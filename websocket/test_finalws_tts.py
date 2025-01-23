import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import asyncio
import websockets
import multiprocessing
import pyaudio
import torch
from final_test.WebSocket.websocket import run_ws_server
from final_test.func import tts_process_func


SOUND_LEVEL = 10
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TIMEOUT_SEC = 0.3

def main():
    stop_event = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    uncheck_audio_queue = multiprocessing.Queue()
    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    try:
        ws_process = multiprocessing.Process(
            target=run_ws_server,
            args=(uncheck_audio_queue, asr_output_queue, llm_output_queue, audio_queue)
        )
        tts_process = multiprocessing.Process(target=tts_process_func, args=(stop_event, llm_output_queue, speaking_event, audio_queue))
        
        ws_process.start()
        tts_process.start()
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        ws_process.join()
        tts_process.join()
        ws_process.close()
        tts_process.close()

        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
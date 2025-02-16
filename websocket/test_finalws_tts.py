import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import asyncio
import websockets
import multiprocessing
import pyaudio
import torch
from final.WebSocket.websocket import run_ws_server
from final.func import tts_process_func

""" 
asr_output_queue replace llm_output_queue here since it is only testing tts 
llm_output_queue_ws for store the text of llm output and send back to the client
"""

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
    asr_output_queue_ws = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    try:
        ws_process = multiprocessing.Process(
            target=run_ws_server,
            args=(uncheck_audio_queue, asr_output_queue, asr_output_queue_ws, llm_output_queue_ws, audio_queue)
        )
        tts_process = multiprocessing.Process(
            target=tts_process_func, 
            args=(
                stop_event, 
                speaking_event, 
                asr_output_queue, # Normally it should be llm_output_queue, but llm is not loaded, only tts is tested
                audio_queue,
            )
        )
        
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

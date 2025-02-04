import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import asyncio
import websockets
import multiprocessing
import pyaudio
import torch
from final.WebSocket.websocket import run_ws_server
from final.func import asr_process_func_ws

""" 
asr_output_queue replace llm_output_queue here since it is only testing asr 
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
    is_user_talking = multiprocessing.Event()

    uncheck_audio_queue = multiprocessing.Queue()
    asr_output_queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    try:
        ws_process = multiprocessing.Process(
            target=run_ws_server, 
            args=(uncheck_audio_queue, asr_output_queue, asr_output_queue, audio_queue) # The third args should be llm_output_queue_ws, but now testing asr output
        )
        asr_process = multiprocessing.Process(
            target=asr_process_func_ws, 
            args=(
                stop_event, 
                uncheck_audio_queue, 
                asr_output_queue,
                is_user_talking, 
            )
        )
        
        ws_process.start()
        asr_process.start()

    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        ws_process.join()
        asr_process.join()
        ws_process.close()
        asr_process.close()

        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()

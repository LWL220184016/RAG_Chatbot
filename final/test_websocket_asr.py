import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from final.func import asr_process_func_ws
from WebSocket.websocket import run_ws_server

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

WS_HOST = "0.0.0.0"
WS_PORT = 6789

def main():
    is_asr_ready_event = multiprocessing.Event()
    is_llm_ready_event = multiprocessing.Event()
    is_tts_ready_event = multiprocessing.Event()
    is_llm_ready_event.set()
    is_tts_ready_event.set()

    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()

    client_audio_queue = multiprocessing.Queue()
    asr_output_queue = multiprocessing.Queue()
    asr_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the user said
    llm_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the llm said
    audio_queue = multiprocessing.Queue()

    try:
        # asr_output_queue for user text input
        ws_process = multiprocessing.Process(
            target=run_ws_server, 
            args=(
                WS_HOST, 
                WS_PORT, 
                is_asr_ready_event, 
                is_llm_ready_event, 
                is_tts_ready_event, 
                client_audio_queue, 
                asr_output_queue, 
                asr_output_queue_ws, 
                llm_output_queue_ws, 
                audio_queue, 
            )
        )
        asr_process = multiprocessing.Process(
            target=asr_process_func_ws, 
            args=(
                is_user_talking, 
                stop_event, 
                is_asr_ready_event, 
                client_audio_queue, 
                asr_output_queue, 
                asr_output_queue_ws, 
                None, 
                False, 
            ) 
        )
        
        ws_process.start()
        asr_process.start()
        while not is_asr_ready_event.is_set():
            time.sleep(0.1)
        
        while not stop_event.is_set():
            while not asr_output_queue.empty():
                output = asr_output_queue.get_nowait()
                print(f"\n\033[38;5;208m🔍 ASR: {output}\033[0m")  # 橙色高亮 (256-color)
            time.sleep(0.1)

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
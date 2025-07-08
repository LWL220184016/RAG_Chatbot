import multiprocessing
import torch
import time

from WebSocket.websocket import run_ws_server
from final.func import llm_process_func_ws

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687" export NEO4J_USERNAME="username" export NEO4J_PASSWORD="password"
# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

WS_HOST = "0.0.0.0"
WS_PORT = 6789

def main():
    is_asr_ready_event = multiprocessing.Event()
    is_llm_ready_event = multiprocessing.Event()
    is_tts_ready_event = multiprocessing.Event()
    is_asr_ready_event.set()
    is_tts_ready_event.set()
    
    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    client_audio_queue = multiprocessing.Queue()
    asr_output_queue = multiprocessing.Queue()
    asr_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the user said
    llm_output_queue = multiprocessing.Queue()
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
                is_user_talking, 
                client_audio_queue, 
                asr_output_queue, 
                asr_output_queue_ws, 
                llm_output_queue_ws, 
                audio_queue, 
            )
        )
        llm_process = multiprocessing.Process(
            target=llm_process_func_ws, 
            args=(
                is_user_talking, 
                stop_event, 
                speaking_event, 
                is_llm_ready_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws,
                None, 
                True, 
            )
        )
        
        ws_process.start()
        llm_process.start()

        while not stop_event.is_set():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        ws_process.join()
        llm_process.join()
        ws_process.close()
        llm_process.close()

        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
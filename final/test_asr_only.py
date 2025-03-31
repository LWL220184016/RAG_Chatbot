import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from func import asr_process_func

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    
    is_asr_ready_event = multiprocessing.Event()

    is_user_talking = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()

    try:
        asr_process = multiprocessing.Process( 
            # target=asr_process_func, 
            target=asr_process_func, 
            args=( 
                is_user_talking, 
                stop_event, 
                is_asr_ready_event, 
                asr_output_queue, 
                None, 
                True, 
            ) 
        ) 
        
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
        asr_process.join()
        asr_process.close()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    main()
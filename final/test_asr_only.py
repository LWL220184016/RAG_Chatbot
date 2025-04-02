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

    user_start_speek_time = None # for 計算識別所需時間

    try:
        asr_process = multiprocessing.Process( 
            target=asr_process_func, 
            args=( 
                is_user_talking, 
                stop_event, 
                is_asr_ready_event, 
                asr_output_queue, 
                "NeMo", # asr_class = "faster_whisper", "NeMo"
                None, # ap = Audio_Processor
                True, # stream = True, False
            ) 
        ) 
        
        asr_process.start()
        while not is_asr_ready_event.is_set():
            time.sleep(0.1)
        
        while not stop_event.is_set():
            if is_user_talking.is_set() and user_start_speek_time == None:
                user_start_speek_time = time.time()

            while not asr_output_queue.empty():
                output = asr_output_queue.get_nowait()
                print(f"\n\033[38;5;208m🔍 ASR: {output}\033[0m")  # 橙色高亮 (256-color)
                print("Time use: ", (time.time() - user_start_speek_time))
                user_start_speek_time = None
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
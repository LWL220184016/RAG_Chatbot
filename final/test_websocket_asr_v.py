import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from veriables import (
    is_asr_ready_event,
    stop_event,
    ws_process,
    asr_process,
    asr_output_queue,
)

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    try:
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
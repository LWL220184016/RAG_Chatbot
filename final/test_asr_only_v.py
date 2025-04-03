import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from veriables import (
    is_asr_ready_event, 
    is_user_talking, 
    stop_event, 
    asr_output_queue, 
    asr_process,
    cleanup_processes
)

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    # Initialize user_start_speek_time variable
    user_start_speek_time = None
    
    try:
        asr_process.start()
        print("Waiting for ASR to be ready...")
        while not is_asr_ready_event.is_set():
            time.sleep(0.1)
        
        print("ASR ready! Listening for speech...")
        
        while not stop_event.is_set():
            if is_user_talking.is_set() and user_start_speek_time is None:
                user_start_speek_time = time.time()
                print("User started speaking...")

            while not asr_output_queue.empty():
                output = asr_output_queue.get_nowait()
                if output:  # Only process non-empty outputs
                    print(f"\n\033[38;5;208m🔍 ASR: {output}\033[0m")  # 橙色高亮 (256-color)
                    if user_start_speek_time is not None:
                        print(f"Processing time: {time.time() - user_start_speek_time:.2f} seconds")
                        user_start_speek_time = None
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down...")
    finally:
        # Clean up resources properly
        stop_event.set()
        if asr_process.is_alive():
            asr_process.join(timeout=3.0)
            if asr_process.is_alive():
                asr_process.terminate()
        asr_process.close()
        
        torch.cuda.ipc_collect()
        print("Program terminated cleanly")


if __name__ == "__main__":
    main()
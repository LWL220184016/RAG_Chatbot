import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from final.chatbot_config import Chatbot_config


# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    chatbot_config = Chatbot_config()

    # Initialize user_start_speek_time variable
    user_start_speek_time = None
    
    try:
        chatbot_config.asr_process.start()
        print("Waiting for ASR to be ready...")
        while not chatbot_config.is_asr_ready_event.is_set():
            time.sleep(0.1)
        
        print("ASR ready! Listening for speech...")
        
        while not chatbot_config.stop_event.is_set():
            if chatbot_config.is_user_talking.is_set() and user_start_speek_time is None:
                user_start_speek_time = time.time()
                print("User started speaking...")

            while not chatbot_config.asr_output_queue.empty():
                output = chatbot_config.asr_output_queue.get_nowait()
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
        chatbot_config.stop_event.set()
        if chatbot_config.asr_process.is_alive():
            chatbot_config.asr_process.join(timeout=3.0)
            if chatbot_config.asr_process.is_alive():
                chatbot_config.asr_process.terminate()
        chatbot_config.asr_process.close()
        
        torch.cuda.ipc_collect()
        print("Program terminated cleanly")


if __name__ == "__main__":
    main()
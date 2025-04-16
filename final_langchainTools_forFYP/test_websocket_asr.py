import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from chatbot_config import Chatbot_config

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    try:
        chatbot_config = Chatbot_config()
        chatbot_config.is_llm_ready_event.set()
        chatbot_config.is_tts_ready_event.set()

        chatbot_config.ws_process.start()
        chatbot_config.asr_process_ws.start()
        while not chatbot_config.is_asr_ready_event.is_set():
            time.sleep(0.1)
        
        while not chatbot_config.stop_event.is_set():
            while not chatbot_config.asr_output_queue.empty():
                output = chatbot_config.asr_output_queue.get_nowait()
                print(f"\n\033[38;5;208m🔍 ASR: {output}\033[0m")  # 橙色高亮 (256-color)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        chatbot_config.stop_event.set()
        chatbot_config.ws_process.join()
        chatbot_config.asr_process_ws.join()
        chatbot_config.ws_process.close()
        chatbot_config.asr_process_ws.close()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    main()
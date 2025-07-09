import torch
import time
import threading
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from chatbot_config import Chatbot_config
from langchain_community.agent_toolkits.load_tools import load_tools

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def get_user_msg(stop_event, asr_output_queue):
    while not stop_event.is_set():
        user_input = input()
        asr_output_queue.put(user_input)

        time.sleep(0.1)
    print("get_user_msg_thread stopped")

def main():
    chatbot_config = Chatbot_config()

    try:
        chatbot_config.llm_process_ws.start()

        get_user_msg_thread = threading.Thread(target=get_user_msg, args=(chatbot_config.stop_event, chatbot_config.asr_output_queue))
        
        print("Waiting for LLM to be ready...")
        while not chatbot_config.is_llm_ready_event.is_set():
            time.sleep(0.1)
        
        get_user_msg_thread.start()
        print("LLM ready! Waiting for user message...")
        
        while not chatbot_config.stop_event.is_set():
            while not chatbot_config.llm_output_queue.empty():
                output = chatbot_config.llm_output_queue.get_nowait()
                if output:  # Only process non-empty outputs
                    print(f"\n\033[38;5;208mLLM: {output}\033]0m")  # 橙色高亮 (256-color)

            time.sleep(0.1)

            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        chatbot_config.stop_event.set()
        get_user_msg_thread.join()
        chatbot_config.llm_process_ws.join()
        chatbot_config.llm_process_ws.close()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    main()
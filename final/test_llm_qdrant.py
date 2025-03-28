import torch
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from LLM.prompt_template import get_langchain_PromptTemplate_Chinese2
from func_fyp import llm_process_func_ws
from langchain_community.agent_toolkits.load_tools import load_tools

# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    
    is_llm_ready_event = multiprocessing.Event()

    is_user_talking = multiprocessing.Event()
    stop_event = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()

    prompt_template = get_langchain_PromptTemplate_Chinese2()
    try:
        llm_process = multiprocessing.Process(
            # target=llm_model_process_func_ws, 
            target=llm_process_func_ws, 
            args=( 
                is_user_talking, 
                stop_event, 
                speaking_event, 
                is_llm_ready_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws, 
                prompt_template, 
                "google", # llm_class
                True, # use_agent
                "qdrant", # use_database
            )
        )
        
        llm_process.start()
        asr_output_queue.put("Hello")

        while not is_llm_ready_event.is_set():
            time.sleep(0.1)

        while not stop_event.is_set():
            user_input = input()
            if user_input == "show":
                while not llm_output_queue.empty():
                    output = llm_output_queue.get_nowait()
                    print(f"\n\033[38;5;208m🔍 LLM: {output}\033[0m")  # 橙色高亮 (256-color)
            else:
                asr_output_queue.put(user_input)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        llm_process.join()
        llm_process.close()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    main()
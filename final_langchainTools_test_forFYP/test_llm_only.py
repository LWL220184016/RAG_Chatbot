import multiprocessing
import torch

# from LLM.llm_ollama import LLM_Ollama as LLM
from LLM.llm_google import LLM_Google as LLM
# from LLM.llm_transformers import LLM_Transformers as LLM
from LLM.prompt_template import get_langchain_PromptTemplate_Chinese2
from func_fyp import llm_agent_process_func_ws
from langchain_community.agent_toolkits.load_tools import load_tools
from Tools.duckduckgo_searching import duckduckgo_search

from Data_Storage.qdrant import Qdrant_Handler
from Data_Storage.embedding_model.embedder import Embedder
# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    tools=[duckduckgo_search]
    
    is_llm_ready_event = multiprocessing.Event()

    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()

    embedder = Embedder()
    llm = LLM( 
        # model_name="deepseek-r1_14b_FYP4", 
        # torch_dtype=torch.float32, 
        # device="cuda:0", 
        is_user_talking=is_user_talking, 
        stop_event=stop_event, 
        speaking_event=speaking_event, 
        user_input_queue=asr_output_queue, 
        llm_output_queue=llm_output_queue, 
        llm_output_queue_ws=llm_output_queue_ws, 
        tools=tools, 
        database=Qdrant_Handler(embedder=embedder), 嘗試吧 embedder model 塞到另一個進程或者線程來解決 RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    ) 

    prompt_template = get_langchain_PromptTemplate_Chinese2()

    try:
        llm_process = multiprocessing.Process(
            target=llm_agent_process_func_ws, 
            args=(
                stop_event, 
                is_user_talking, 
                speaking_event, 
                is_llm_ready_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws,
                prompt_template,
                llm,
            )
        )
        
        llm_process.start()


        asr_output_queue.put("Hello")
        while not stop_event.is_set():
            user_input = input("User: ")
            if user_input == "show":
                while not llm_output_queue.empty():
                    output = llm_output_queue.get_nowait()
                    print(output)
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
    # multiprocessing.set_start_method('spawn')
    main()
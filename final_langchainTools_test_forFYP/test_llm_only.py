import multiprocessing
import torch

# from LLM.llm_ollama import LLM_Ollama as LLM
from LLM.llm_google import LLM_Google as LLM
# from LLM.llm_transformers import LLM_Transformers as LLM
from LLM.prompt_template import get_langchain_PromptTemplate_Chinese2
# from RAG.graph_rag import Graph_RAG
from func_fyp import llm_agent_process_func_ws
from langchain_community.agent_toolkits.load_tools import load_tools
from Tools.duckduckgo_searching import duckduckgo_search

def main():
    tools=[duckduckgo_search]
    
    is_llm_ready_event = multiprocessing.Event()

    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()

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
        tools=tools
    )

    # rag = Graph_RAG()
    rag = None
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
                rag,
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
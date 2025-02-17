import multiprocessing
import torch
import threading

# from LLM.llm_ollama import LLM
from LLM.llm_google import LLM
from LLM.prompt_template import get_langchain_PromptTemplate
# from RAG.graph_rag import Graph_RAG
from func_fyp import llm_process_func_ws
from langchain_community.agent_toolkits.load_tools import load_tools
from Tools.duckduckgo_searching import duckduckgo_search

def main():
    tools=[duckduckgo_search]

    llm = LLM(
        is_user_talking=is_user_talking, 
        stop_event=stop_event, 
        speaking_event=speaking_event, 
        llm_output_queue=llm_output_queue,
        tools=tools
    )

    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()

    # rag = Graph_RAG()
    prompt_template = get_langchain_PromptTemplate()

    try:
        llm_process = multiprocessing.Process(
            target=llm_process_func_ws, 
            args=(
                llm,
                stop_event, 
                is_user_talking, 
                speaking_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws,
                prompt_template,
                # rag,
            )
        )
        
        llm_process.start()



        while not stop_event.is_set():
            user_input = input("User: ")
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
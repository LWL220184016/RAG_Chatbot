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

def show_queue_len(stop_event, queue):
    while not stop_event.is_set():
        print("queue size: ", queue.qsize())
        llm_msg = queue.get()
        print("queue.get: ", llm_msg)

def main():
    tools=[duckduckgo_search]
    
    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()

    llm = LLM(
        is_user_talking=is_user_talking, 
        stop_event=stop_event, 
        speaking_event=speaking_event, 
        llm_output_queue=llm_output_queue,
        tools=tools
    )

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

        show_queue_len_thread = threading.Thread(target=show_queue_len, args=(stop_event, llm_output_queue,))
        show_queue_len_thread.start()


        while not stop_event.is_set():
            user_input = input("User: ")
            asr_output_queue.put(user_input)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        llm_process.join()
        llm_process.close()
        show_queue_len_thread.join()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    main()
import multiprocessing
import torch
import time

# from LLM.llm_ollama import LLM
from LLM.llm_google import LLM
from LLM.prompt_template import get_langchain_PromptTemplate
# from RAG.graph_rag import Graph_RAG
from WebSocket.websocket import run_ws_server
from func_fyp import llm_process_func_ws
from Tools.duckduckgo_searching import duckduckgo_search

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687" export NEO4J_USERNAME="username" export NEO4J_PASSWORD="password"
WS_HOST = "0.0.0.0"
WS_PORT = 6789

def main():
    tools=[duckduckgo_search]

    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    client_audio_queue = multiprocessing.Queue()
    asr_output_queue = multiprocessing.Queue()
    asr_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the user said
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the llm said
    audio_queue = multiprocessing.Queue()

    llm = LLM(
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
    prompt_template = get_langchain_PromptTemplate()

    try:
        # asr_output_queue for user text input
        ws_process = multiprocessing.Process(
            target=run_ws_server, 
            args=(
                WS_HOST, 
                WS_PORT, 
                client_audio_queue, 
                asr_output_queue, 
                asr_output_queue_ws, 
                llm_output_queue_ws, 
                audio_queue, 
            )
        )
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
                rag, 
            )
        )

        
        ws_process.start()
        llm_process.start()

        while not stop_event.is_set():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        ws_process.join()
        llm_process.join()
        ws_process.close()
        llm_process.close()

        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
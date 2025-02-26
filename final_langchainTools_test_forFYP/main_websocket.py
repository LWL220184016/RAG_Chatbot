import pyaudio
import sounddevice as sd
import multiprocessing
import torch
import time

# from ASR.asr import ASR
from final_langchainTools_test_forFYP.LLM.llm_ollama import LLM_Ollama as LLM
from TTS.tts_transformers import TTS
# from RAG.graph_rag import Graph_RAG
from LLM.prompt_template import get_langchain_PromptTemplate_Chinese2
from WebSocket.websocket import run_ws_server
from func_fyp import asr_process_func_ws, llm_agent_process_func_ws, tts_process_func
from Tools.duckduckgo_searching import duckduckgo_search

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687" export NEO4J_USERNAME="username" export NEO4J_PASSWORD="password"
ws_host = "localhost"
ws_port = 6789

SOUND_LEVEL = 10
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TIMEOUT_SEC = 0.3

def main():
    tools=[duckduckgo_search]

    is_asr_ready_event = multiprocessing.Event()
    is_llm_ready_event = multiprocessing.Event()
    is_tts_ready_event = multiprocessing.Event()

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
        model_name="deepseek-r1_14b_FYP4", 
        is_user_talking=is_user_talking, 
        stop_event=stop_event, 
        speaking_event=speaking_event, 
        user_input_queue=asr_output_queue, 
        llm_output_queue=llm_output_queue, 
        llm_output_queue_ws=llm_output_queue_ws, 
        tools=tools, 
    )

    # rag = Graph_RAG()
    rag = None
    prompt_template = get_langchain_PromptTemplate_Chinese2()

    try:
        # asr_output_queue for user text input
        ws_process = multiprocessing.Process(
            target=run_ws_server, 
            args=(
                ws_host, 
                ws_port, 
                is_asr_ready_event, 
                is_llm_ready_event, 
                is_tts_ready_event, 
                client_audio_queue, 
                asr_output_queue, 
                asr_output_queue_ws, 
                llm_output_queue_ws, 
                audio_queue, 
            )
        )
        asr_process = multiprocessing.Process(
            target=asr_process_func_ws, 
            args=(
                stop_event, 
                is_user_talking, 
                is_asr_ready_event, 
                client_audio_queue, 
                asr_output_queue, 
                asr_output_queue_ws, 
            )
        )
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
        tts_process = multiprocessing.Process(
            target=tts_process_func, 
            args=(
                stop_event, 
                speaking_event, 
                is_tts_ready_event, 
                llm_output_queue, 
                audio_queue, 
            )
        )

        
        ws_process.start()
        asr_process.start()
        llm_process.start()
        tts_process.start()

        while not stop_event.is_set():
            # while speaking_event.is_set() or not audio_queue.empty():
            #     audio_chunk = audio_queue.get()
            #     sd.play(audio_chunk, samplerate=16000, blocking=False)
            #     while sd.get_stream().active:
            #         if is_user_talking.is_set():
            #             sd.stop()
            #             if not audio_queue.empty():
            #                 audio_chunk = audio_queue.get()
            #             break
                    # time.sleep(0.01)


            time.sleep(1)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        ws_process.join()
        asr_process.join()
        llm_process.join()
        tts_process.join()
        ws_process.close()
        asr_process.close()
        llm_process.close()
        tts_process.close()

        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
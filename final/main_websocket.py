import pyaudio
import sounddevice as sd
import multiprocessing
import torch
import time

# from ASR.asr import ASR
from LLM.prompt_template import Message
from RAG.graph_rag import Graph_RAG
from WebSocket.websocket import run_ws_server
from func import asr_process_func_ws, llm_process_func, tts_process_func_ws

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687"
# export NEO4J_USERNAME="username"
# export NEO4J_PASSWORD="password"

SOUND_LEVEL = 10
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TIMEOUT_SEC = 0.3

def main():
    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    uncheck_audio_queue = multiprocessing.Queue()
    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    rag = Graph_RAG()
    user_message = Message("best friend1")
    llm_message = Message("best friend2")

    try:
        # asr_output_queue for user text input
        ws_process = multiprocessing.Process(
            target=run_ws_server, 
            args=(uncheck_audio_queue, asr_output_queue, llm_output_queue_ws, audio_queue)
        )
        asr_process = multiprocessing.Process(target=asr_process_func_ws, args=(stop_event, uncheck_audio_queue, asr_output_queue, is_user_talking))
        llm_process = multiprocessing.Process(target=llm_process_func, args=(stop_event, is_user_talking, speaking_event, asr_output_queue, llm_output_queue, user_message, llm_message, rag))
        tts_process = multiprocessing.Process(target=tts_process_func_ws, args=(stop_event, llm_output_queue, llm_output_queue_ws, audio_queue, speaking_event))
        
        
        ws_process.start()
        asr_process.start()
        llm_process.start()
        tts_process.start()

        while not stop_event.is_set():
            while speaking_event.is_set() or not audio_queue.empty():
                audio_chunk = audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=False)
                while sd.get_stream().active:
                    if is_user_talking.is_set():
                        sd.stop()
                        if not audio_queue.empty():
                            audio_chunk = audio_queue.get()
                        break
                    time.sleep(0.01)
            
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
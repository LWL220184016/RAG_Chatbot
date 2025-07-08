import multiprocessing
import torch
import queue
import sounddevice as sd
import time
# from LLM.llm_ollama import LLM_Ollama as LLM
from LLM.llm_google import LLM_Google as LLM
from final.func import llm_process_func_ws, tts_process_func
from langchain_community.agent_toolkits.load_tools import load_tools
from final_langchainTools_forFYP.Tools.duckduckgo import duckduckgo_search

def main():
    is_llm_ready_event = multiprocessing.Event()
    is_tts_ready_event = multiprocessing.Event()
    
    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()


    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    try:
        llm_process = multiprocessing.Process(
            target=llm_process_func_ws, 
            args=( 
                is_user_talking, 
                stop_event, 
                speaking_event, 
                is_llm_ready_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws, 
                "google", # llm_class: "google", "transformers"
                True, # use_agent: True, False
                None, # use_database: None, "qdrant"
            )
        )
        tts_process = multiprocessing.Process(
            target=tts_process_func, 
            args=(
                stop_event, 
                speaking_event, 
                llm_output_queue, 
                audio_queue,
            )
        )
        
        llm_process.start()


        asr_output_queue.put("Hello")
        while not stop_event.is_set():
            user_input = input(prompt="User: ")
            asr_output_queue.put(user_input)


            # while speaking_event.is_set() or not audio_queue.empty():
            #     try:
            #         audio_chunk = audio_queue.get()
            #     except queue.Empty:
            #         continue
            #     sd.play(audio_chunk, samplerate=16000, blocking=False)
            #     while sd.get_stream().active:
            #         if is_user_talking.is_set():
            #             sd.stop()
            #             if not audio_queue.empty():
            #                 audio_chunk = audio_queue.get()
            #             break
            #         time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        llm_process.join()
        llm_process.close()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    main()
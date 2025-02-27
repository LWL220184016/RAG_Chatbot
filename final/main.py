import pyaudio
import threading
import sounddevice as sd
import multiprocessing
import torch
import time
import queue

from ASR.audio_process import Audio_Processer
# from ASR.asr import ASR
from ASR.model_classes.NeMo import ASR
from final.LLM.llm_ollama import LLM_Ollama as LLM
from LLM.prompt_template import Message
from final.TTS.tts_transformers import TTS
from final.func import asr_process_func, llm_model_process_func_ws, tts_process_func

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

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    user_message = Message("best friend1")
    llm_message = Message("best friend2")

    try:
        asr_process = multiprocessing.Process(
            target=asr_process_func, 
            args=(
                stop_event, 
                is_user_talking, 
                asr_output_queue
            )
        )
        llm_process = multiprocessing.Process(
            target=llm_model_process_func_ws, 
            args=(
                stop_event, 
                is_user_talking, 
                speaking_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws, 
                user_message, 
                llm_message, 
            )
        )
        tts_process = multiprocessing.Process(
            target=tts_process_func, 
            args=(
                stop_event, 
                llm_output_queue, 
                speaking_event, 
                audio_queue
            )
        )

        asr_process.start()
        llm_process.start()
        tts_process.start()

        while not stop_event.is_set():
            while speaking_event.is_set() or not audio_queue.empty():
                try:
                    audio_chunk = audio_queue.get()
                except queue.Empty:
                    continue
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
        asr_process.join()
        llm_process.join()
        tts_process.join()
        asr_process.close()
        llm_process.close()
        tts_process.close()

        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
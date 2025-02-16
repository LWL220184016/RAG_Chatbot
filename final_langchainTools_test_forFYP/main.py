import pyaudio
import threading
import sounddevice as sd
import multiprocessing
import torch
import time
import queue

from ASR.audio_process import Audio_Processer
# from ASR.asr import ASR
from ASR.model_classes.NeMo import NeMo_ASR as ASR
from LLM.llm import LLM
from LLM.prompt_template import Message
from TTS.tts import TTS
from RAG.graph_rag import Graph_RAG

SOUND_LEVEL = 10
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TIMEOUT_SEC = 0.3

def asr_process_func(stop_event, asr_output_queue, is_user_talking):
    try:
        ap = Audio_Processer(
            chunk=CHUNK, 
            format=FORMAT, 
            channels=CHANNELS, 
            rate=RATE, 
            is_user_talking=is_user_talking, 
            stop_event=stop_event
        )
        get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
        check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL, TIMEOUT_SEC))
        get_audio_thread.start()
        check_audio_thread.start()
        asr = ASR(stop_event=stop_event, ap=ap, asr_output_queue=asr_output_queue)
        print("asr_process_func asring")
        asr.asr_output()
        print("asr_process_func end")

    except KeyboardInterrupt:
        print("asr_process_func KeyboardInterrupt\n")
        get_audio_thread.join()
        check_audio_thread.join()
        get_audio_thread.close()
        check_audio_thread.close()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        torch.cuda.ipc_collect()

    finally:
        print("asr_process_func finally\n")
        get_audio_thread.join()
        check_audio_thread.join()
        get_audio_thread.close()
        check_audio_thread.close()
        torch.cuda.ipc_collect()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()

def llm_process_func(stop_event, is_user_talking, speaking_event, asr_output_queue, llm_output_queue, user_message, llm_message, rag):
    try:
        llm = LLM(is_user_talking=is_user_talking, stop_event=stop_event, speaking_event=speaking_event, llm_output_queue=llm_output_queue)
        llm.llm_output(asr_output_queue, user_message, llm_message, rag)
    except KeyboardInterrupt:
        print("llm_process_func KeyboardInterrupt\n")
        stop_event.set()
    finally:
        print("llm_process_func finally\n")
        stop_event.set()
        torch.cuda.ipc_collect()

def tts_process_func(stop_event, llm_output_queue, speaking_event, audio_queue):
    try:
        tts = TTS(stop_event=stop_event, audio_queue=audio_queue)
        tts.tts_output(llm_output_queue, speaking_event)
    except KeyboardInterrupt:
        print("tts_process_func KeyboardInterrupt\n")
        stop_event.set()
    finally:
        print("tts_process_func finally\n")
        stop_event.set()
        torch.cuda.ipc_collect()

def main():
    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()

    rag = Graph_RAG()
    user_message = Message("best friend1")
    llm_message = Message("best friend2")

    try:
        asr_process = multiprocessing.Process(target=asr_process_func, args=(stop_event, asr_output_queue, is_user_talking))
        llm_process = multiprocessing.Process(target=llm_process_func, args=(stop_event, is_user_talking, speaking_event, asr_output_queue, llm_output_queue, user_message, llm_message, rag))
        tts_process = multiprocessing.Process(target=tts_process_func, args=(stop_event, llm_output_queue, speaking_event, audio_queue))
        
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
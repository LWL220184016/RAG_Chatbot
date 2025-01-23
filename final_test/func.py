import pyaudio
import threading
from ASR.audio_process import Audio_Processer
# from ASR.asr import ASR
from ASR.model_classes.NeMo import NeMo_ASR as ASR
from LLM.llm import LLM
from LLM.prompt_template import Message
from TTS.tts import TTS
from RAG.graph_rag import Graph_RAG
import torch

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

def asr_process_func_ws(stop_event, uncheck_audio_queue, asr_output_queue, is_user_talking):
    try:
        ap = Audio_Processer(
            chunk=CHUNK, 
            format=FORMAT, 
            channels=CHANNELS, 
            rate=RATE, 
            audio_unchecked_queue=uncheck_audio_queue,
            is_user_talking=is_user_talking, 
            stop_event=stop_event
        )
        check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL, TIMEOUT_SEC))
        check_audio_thread.start()
        asr = ASR(stop_event=stop_event, ap=ap, asr_output_queue=asr_output_queue)
        print("asr_process_func asring")
        asr.asr_output()
        print("asr_process_func end")

    except KeyboardInterrupt:
        print("asr_process_func KeyboardInterrupt\n")
        check_audio_thread.join()
        check_audio_thread.close()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        torch.cuda.ipc_collect()

    finally:
        print("asr_process_func finally\n")
        check_audio_thread.join()
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
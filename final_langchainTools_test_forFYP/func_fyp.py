import threading
import torch
import traceback
import multiprocessing.queues

# from ASR.audio_process import Audio_Processer
# from ASR.asr import ASR
# from ASR.model_classes.NeMo import ASR
from LLM.llm_ollama import LLM
# from TTS.tts_transformers import TTS
# from RAG.graph_rag import Graph_RAG


# def asr_process_func(
#         ap: Audio_Processer, 
#         stop_event: threading.Event, 
#         is_user_talking: threading.Event,
#         asr_output_queue: multiprocessing.Queue, 
#         ):
#     import pyaudio
#     SOUND_LEVEL = 10
#     CHUNK = 512
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 16000
#     TIMEOUT_SEC = 0.3

#     try:
#         if ap is None:
#             ap = Audio_Processer(
#                 chunk=CHUNK, 
#                 format=FORMAT, 
#                 channels=CHANNELS, 
#                 rate=RATE, 
#                 is_user_talking=is_user_talking, 
#                 stop_event=stop_event
#             )
#         get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
#         check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL, TIMEOUT_SEC))
#         get_audio_thread.start()
#         check_audio_thread.start()
#         asr = ASR(stop_event=stop_event, ap=ap, asr_output_queue=asr_output_queue)
#         print("asr_process_func asring")
#         asr.asr_output()
#         print("asr_process_func end")

#     except KeyboardInterrupt:
#         print("asr_process_func KeyboardInterrupt\n")
#         get_audio_thread.join()
#         check_audio_thread.join()
#         get_audio_thread.close()
#         check_audio_thread.close()
#         ap.stream.stop_stream()
#         ap.stream.close()
#         ap.p.terminate()
#         torch.cuda.ipc_collect()

#     finally:
#         print("asr_process_func finally\n")
#         get_audio_thread.join()
#         check_audio_thread.join()
#         get_audio_thread.close()
#         check_audio_thread.close()
#         torch.cuda.ipc_collect()
#         ap.stream.stop_stream()
#         ap.stream.close()
#         ap.p.terminate()

# def asr_process_func_ws(
#         ap: Audio_Processer, 
#         stop_event: threading.Event, 
#         is_user_talking: threading.Event,
#         uncheck_audio_queue: multiprocessing.Queue, 
#         asr_output_queue: multiprocessing.Queue, 
#         asr_output_queue_ws: multiprocessing.Queue, 
#         ):
#     import pyaudio
#     CHUNK = 512
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 16000

#     try:
#         if ap is None:
#             ap = Audio_Processer(
#                 chunk=CHUNK, 
#                 format=FORMAT, 
#                 channels=CHANNELS, 
#                 rate=RATE, 
#                 audio_checked_queue=uncheck_audio_queue,
#                 startStream=False,
#                 is_user_talking=is_user_talking, 
#                 stop_event=stop_event,
#             )
#         asr = ASR(stop_event=stop_event, ap=ap, asr_output_queue=asr_output_queue)
#         asr.asr_output_ws(asr_output_queue_ws)
#         print("asr_process_func end")

#     except KeyboardInterrupt:
#         print("asr_process_func KeyboardInterrupt\n")
#         ap.p.terminate()
#         torch.cuda.ipc_collect()
    
#     except Exception as e:
#         print("捕获异常：", e)
#         print("完整的错误信息：")
#         traceback.print_exc()

#     finally:
#         print("asr_process_func finally\n")
#         torch.cuda.ipc_collect()
#         ap.p.terminate()

def llm_process_func_ws(
        llm: LLM,
        stop_event: threading.Event, 
        is_user_talking: threading.Event, 
        speaking_event: threading.Event, 
        asr_output_queue: multiprocessing.Queue, 
        llm_output_queue: multiprocessing.Queue, 
        llm_output_queue_ws: multiprocessing.Queue, 
        prompt_template,
        rag = None,
        ):
    
    try:
        if llm is None:
            llm = LLM(
                is_user_talking=is_user_talking, 
                stop_event=stop_event, 
                speaking_event=speaking_event, 
                llm_output_queue=llm_output_queue
            )
        llm.llm_output_ws(asr_output_queue, llm_output_queue_ws, prompt_template, rag)
    except KeyboardInterrupt:
        print("llm_process_func KeyboardInterrupt\n")
        stop_event.set()
    finally:
        print("llm_process_func finally\n")
        stop_event.set()
        torch.cuda.ipc_collect()

# def tts_process_func(
#         tts: TTS, 
#         stop_event: threading.Event, 
#         speaking_event: threading.Event, 
#         llm_output_queue: multiprocessing.Queue, 
#         audio_queue: multiprocessing.Queue
#     ):
#     try:
#         if tts is None:
#             tts = TTS(stop_event=stop_event, audio_queue=audio_queue)
#         tts.tts_output(llm_output_queue, speaking_event)
#     except KeyboardInterrupt:
#         print("tts_process_func KeyboardInterrupt\n")
#         stop_event.set()
#     finally:
#         print("tts_process_func finally\n")
#         stop_event.set()
#         torch.cuda.ipc_collect()

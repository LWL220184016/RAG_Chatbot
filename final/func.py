import queue
import threading
import torch
import traceback

def asr_process_func(
        is_user_talking: threading.Event,
        stop_event: threading.Event, 
        is_asr_ready_event: threading.Event,
        asr_output_queue: queue, 
        ap = None, 
        streaming=False, 
    ):
    """
    ap: Audio_Processer
    """

    import pyaudio
    from ASR.audio_process import Audio_Processer
    # from ASR.model_classes.faster_whisper import ASR
    from ASR.model_classes.NeMo import ASR

    SOUND_LEVEL = 5
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    TIMEOUT_SEC = 0.3

    try:
        if ap is None:
            ap = Audio_Processer( 
                chunk=CHUNK, 
                format=FORMAT, 
                channels=CHANNELS, 
                rate=RATE, 
                is_user_talking=is_user_talking, 
                stop_event=stop_event, 
            ) 
        get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
        if streaming:
            check_audio_thread = threading.Thread(target=ap.detect_sound_not_extend, args=(SOUND_LEVEL, TIMEOUT_SEC))
        else:
            check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL, TIMEOUT_SEC))
        get_audio_thread.start()
        check_audio_thread.start()
        asr = ASR( 
            stop_event=stop_event, 
            is_user_talking=is_user_talking, 
            ap=ap, 
            asr_output_queue=asr_output_queue, 
            streaming=streaming, 
        ) 
        print("asr_process_func asring")
        asr.asr_output(is_asr_ready_event)
    except KeyboardInterrupt:
        print("asr_process_func KeyboardInterrupt\n")
        get_audio_thread.join()
        check_audio_thread.join()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        torch.cuda.ipc_collect()

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        print("asr_process_func finally\n")
        get_audio_thread.join()
        check_audio_thread.join()
        torch.cuda.ipc_collect()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()

def asr_process_func_ws(
        is_user_talking: threading.Event, 
        stop_event: threading.Event, 
        is_asr_ready_event: threading.Event, 
        uncheck_audio_queue: queue, 
        asr_output_queue: queue, 
        asr_output_queue_ws: queue, 
        ap = None, 
        streaming=False, 
        chunk=4096,
    ): 
    """
    ap: Audio_Processer
    """

    import pyaudio
    from ASR.audio_process import Audio_Processer
    # from ASR.model_classes.faster_whisper import ASR
    from ASR.model_classes.NeMo import ASR
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    try:
        if ap is None:
            ap = Audio_Processer(
                chunk=chunk, 
                format=FORMAT, 
                channels=CHANNELS, 
                rate=RATE, 
                audio_checked_queue=uncheck_audio_queue,
                startStream=False,
                is_user_talking=is_user_talking, 
                stop_event=stop_event,
            )
        asr = ASR(
            device="cuda:0",
            ap=ap, 
            stop_event=stop_event, 
            is_user_talking=is_user_talking, 
            asr_output_queue=asr_output_queue, 
            streaming=streaming, 
        )
        asr.asr_output_ws(is_asr_ready_event, asr_output_queue_ws)
        print("asr_process_func end")

    except KeyboardInterrupt:
        print("asr_process_func KeyboardInterrupt\n")
        ap.p.terminate()
        torch.cuda.ipc_collect()
    
    except Exception as e:
        print("捕获异常：", e)
        print("完整的错误信息：")
        traceback.print_exc()

    finally:
        print("asr_process_func finally\n")
        torch.cuda.ipc_collect()
        ap.p.terminate()

def llm_process_func_ws( 
        is_user_talking: threading.Event, 
        stop_event: threading.Event, 
        speaking_event: threading.Event, 
        is_llm_ready_event: threading.Event, 
        asr_output_queue: queue, 
        llm_output_queue: queue, 
        llm_output_queue_ws: queue, 
        llm_class = "google", 
        use_agent = False, 
        use_database = None,
    ):

    # from LLM.llm_transformers import LLM_Transformers as LLM
    # from LLM.llm_google import LLM_Google as LLM
    # from LLM.llm_ollama import LLM_Ollama as LLM
    from Tools.tool import Tools

    LLM = get_llm_class(llm_class)

    if use_database not in [None, "qdrant"]:
        raise ValueError("use_database must be 'none' or 'qdrant'")
    elif use_database == "qdrant":
        from Data_Storage.qdrant import Qdrant_Handler as Database
        database = Database()
    else:
        database = None
    
    Tool = Tools(database_qdrant=database)
    tools = [
        Tool.duckduckgo_search, 
        Tool.querying_qdrant, 
        Tool.get_current_dateTime
    ]

    llm = LLM( 
        # model_name="deepseek-r1_14b_FYP4", 
        # torch_dtype=torch.float32, 
        # device="cuda:0", 
        is_user_talking=is_user_talking, 
        stop_event=stop_event, 
        speaking_event=speaking_event, 
        user_input_queue=asr_output_queue, 
        llm_output_queue=llm_output_queue, 
        llm_output_queue_ws=llm_output_queue_ws, 
        tools=tools, 
        database=database, 
    ) 
    try:
        if use_agent:
            llm.langchain_agent_output_ws(
                is_llm_ready_event=is_llm_ready_event, 
            )
        else:
            llm.llm_output_ws(
                is_llm_ready_event=is_llm_ready_event, 
            )
    except KeyboardInterrupt:
        print("llm_process_func KeyboardInterrupt\n")
        stop_event.set()
    finally:
        print("llm_process_func finally\n")
        stop_event.set()
        torch.cuda.ipc_collect()

def tts_process_func(
        stop_event: threading.Event, 
        speaking_event: threading.Event, 
        is_tts_ready_event: threading.Event,
        llm_output_queue: queue, 
        audio_queue: queue, 
        tts = None, 
    ):
    from TTS.tts_transformers import TTS
    
    try:
        if tts is None:
            tts = TTS(stop_event=stop_event, audio_queue=audio_queue)
        tts.tts_output(speaking_event, is_tts_ready_event, llm_output_queue)
    except KeyboardInterrupt:
        print("tts_process_func KeyboardInterrupt\n")
        stop_event.set()
    finally:
        print("tts_process_func finally\n")
        stop_event.set()
        torch.cuda.ipc_collect()

def get_llm_class(llm_name: str):
    if llm_name == "transformers":
        from LLM.llm_transformers import LLM_Transformers as LLM
    elif llm_name == "google":
        from LLM.llm_google import LLM_Google as LLM
    elif llm_name == "ollama":
        from LLM.llm_ollama import LLM_Ollama as LLM
    else:
        raise ValueError("llm_name must be 'transformers', 'google', or 'ollama'")
    
    return LLM
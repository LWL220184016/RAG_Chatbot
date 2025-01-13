import pyaudio
import threading
import queue
import sounddevice as sd

from faster_whisper import WhisperModel
from ASR.audio_process import Audio_Processer
from LLM.llm import LLM
from LLM.prompt_template import Message
from TTS.tts import TTS
from RAG.rag import RAG

SOUND_LEVEL = 5
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SEC = 1
對話玩第一次后在説話就會卡死，感覺可能是在更新faiss的時候出的問題
if __name__ == "__main__":
    stop_event = threading.Event()
    is_user_talking = threading.Event()
    speaking_event = threading.Event()

    ap = Audio_Processer(chunk=CHUNK, stop_event=stop_event)
    asr_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    llm = LLM(is_user_talking=is_user_talking)
    tts = TTS()
    rag = RAG()
    # sound_device_index = get_input_device(p, "Microphone (MONSTER AIRMARS N3)")

    user_message = Message("best friend1")
    llm_message = Message("best friend2")

    asr_output_queue = queue.Queue()

    try:
        get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
        check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL,))
        llm_thread = threading.Thread(target=llm.llm_output, args=(asr_output_queue, user_message, llm_message, rag))
        tts_thread = threading.Thread(target=tts.tts_output, args=(llm.llm_output_queue, speaking_event))
        get_audio_thread.start()
        check_audio_thread.start()
        llm_thread.start()
        tts_thread.start()

        while not stop_event.is_set():
            print("\nRecording...")
            try:
                audio_data = ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = ap.process_audio2(audio_data=audio_data)
            segments, info = asr_model.transcribe(processed_data, beam_size=5)
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            prompt = ""
            if info.language_probability > 0.5 and (info.language == 'en' or info.language == 'zh'):
                for segment in segments:
                    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    prompt = segment.text
            else:
                continue
            
            print("You: " + prompt)

            # prompt = input("You: ")
            asr_output_queue.put(prompt)
            speaking_event.set()  # Signal that LLM is speaking
            
            while speaking_event.is_set() or not tts.audio_queue.empty():
                audio_chunk = tts.audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=True)
                sd.wait()
                tts.audio_queue.task_done()


    except KeyboardInterrupt:
        stop_event.set()
        get_audio_thread.join()
        check_audio_thread.join()
        llm_thread.join()
        tts_thread.join()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        print("User stopped the program")
import pyaudio
import threading
import queue
import sounddevice as sd

from faster_whisper import WhisperModel
from final.ASR.audio_process import Audio_Processer
from for_test_usage.llm_tts.llm_tts import get_LLM, get_TTS, llm_output, tts_output

SOUND_LEVEL = 5
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SEC = 1

if __name__ == "__main__":
    llm = get_LLM()
    tts_processor, tts_model, vocoder, speaker_embeddings = get_TTS()
    asr_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    ap = Audio_Processer(chunk=CHUNK)
    # sound_device_index = get_input_device(p, "Microphone (MONSTER AIRMARS N3)")

    user_input_queue = queue.Queue()
    llm_output_queue = queue.Queue()
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    speaking_event = threading.Event()

    try:
        get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
        check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL,))
        llm_thread = threading.Thread(target=llm_output, args=(llm, user_input_queue, llm_output_queue))
        tts_thread = threading.Thread(target=tts_output, args=(tts_processor, tts_model, vocoder, speaker_embeddings, llm_output_queue, audio_queue, speaking_event))
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
            if info.language_probability > 0.5:
                for segment in segments:
                    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    prompt = segment.text
            else:
                continue
            
            print("You: " + prompt)

            # prompt = input("You: ")
            user_input_queue.put(prompt)
            speaking_event.set()  # Signal that LLM is speaking
            
            while speaking_event.is_set() or not audio_queue.empty():
                audio_chunk = audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=True)
                sd.wait()
                audio_queue.task_done()

    except KeyboardInterrupt:
        stop_event.set()
        ap.stop_event.set()
        get_audio_thread.join()
        check_audio_thread.join()
        llm_thread.join()
        tts_thread.join()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        print("User stopped the program")
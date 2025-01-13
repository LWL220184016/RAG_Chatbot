import os.path
import sys
# Add the parent folder of the current parent folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))

from faster_whisper import WhisperModel
from final.ASR.audio_process import Audio_Processer
import threading
import queue

if __name__ == "__main__":
    CHUNK = 2048
    SOUND_LEVEL = 20
    model_size = "large-v3"
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    model = WhisperModel(model_size)

    ap = Audio_Processer(chunk=CHUNK)

    get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
    check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL,))

    get_audio_thread.start()
    check_audio_thread.start()
    print("Recording...")
    try:
        while not ap.stop_event.is_set():
            try:
                audio_data = ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = ap.process_audio2(audio_data=audio_data)
            segments, info = model.transcribe(
                processed_data, 
                beam_size=5, 
                language_detection_threshold=0.7
            )
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            
            # if info.language_probability > 0.5:
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            

    except KeyboardInterrupt:
        ap.stop_event.set()
        get_audio_thread.join()
        check_audio_thread.join()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        print("User stopped the program")
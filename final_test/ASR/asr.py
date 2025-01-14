from faster_whisper import WhisperModel
from audio_process import Audio_Processer
import queue
import threading

class ASR():
    def __init__(
        self, 
        model: str="large-v3", 
        device="cuda", 
        compute_type="float16", 
        ap: Audio_Processer=None,
        stop_event = threading.Event()
    ):
    
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.device = device
        self.ap = ap
        self.asr_output_queue = queue.Queue()
        self.stop_event = stop_event

    def asr_output(self):
        while not self.stop_event.is_set():
            audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            processed_data = self.ap.process_audio2(audio_data=audio_data)
            segments, info = self.model.transcribe(processed_data, beam_size=5)
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
            self.asr_output_queue.put(prompt)
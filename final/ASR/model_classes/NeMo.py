from ASR.audio_process import Audio_Processer
import queue
import threading
import nemo.collections.asr as nemo_asr


class NeMo_ASR():
    def __init__(
            self, 
            model: str="nvidia/parakeet-rnnt-1.1b", 
            device="cuda", 
            ap: Audio_Processer=None,
            stop_event: threading.Event = None
        ):
    
        self.model = nemo_asr.models.ASRModel.from_pretrained(model).to(device)
        self.device = device
        self.ap = ap
        self.asr_output_queue = queue.Queue()
        self.stop_event = stop_event

    def asr_output(self):
        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = self.ap.process_audio2(audio_data=audio_data)
            transcriptions = self.model.transcribe(
                # encoded_features[0],
                [processed_data],
                batch_size = 4,
                return_hypotheses = True
            )
            print("Detected language 'en' with probability %f" % (transcriptions))
            # # may need to change the code to check the language_probability
            # prompt = ""
            # if info.language_probability > 0.5 and (info.language == 'en' or info.language == 'zh'):
            #     for segment in segments:
            #         print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            #         prompt = segment.text
            # else:
            #     continue
            
            print("You: " + transcriptions)
            # prompt = input("You: ")
            self.asr_output_queue.put(transcriptions)
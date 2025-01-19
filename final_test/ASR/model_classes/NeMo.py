from ASR.audio_process import Audio_Processer
import queue
import multiprocessing
import nemo.collections.asr as nemo_asr


class NeMo_ASR():
    def __init__(
            self, 
            model: str="nvidia/parakeet-rnnt-1.1b", 
            device="cuda", 
            ap: Audio_Processer=None,
            stop_event = None,
            asr_output_queue: multiprocessing.Queue = None,
        ):
    
        self.model = nemo_asr.models.ASRModel.from_pretrained(model)
        self.model.eval()
        self.device = device
        self.ap = ap
        self.asr_output_queue = asr_output_queue
        self.stop_event = stop_event

    def asr_output(self):
        while not self.stop_event.is_set():
            # try:
            #     audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            # except queue.Empty:
            #     continue
            audio_data = self.ap.audio_checked_queue.get()
            processed_data = self.ap.process_audio2(audio_data=audio_data)
            transcriptions = self.model.transcribe(
                # encoded_features[0],
                processed_data,
                batch_size = 4,
                return_hypotheses = True,
                verbose = False,
            )
            # print("transcriptions: " + str(transcriptions))
            hypothesis = transcriptions[0]
            h = hypothesis[0]
            # print("score: " + str(h.score))
            # print("You: " + h.text)
            if h.score < 50:
                continue
            # prompt = input("You: ")
            self.asr_output_queue.put(h.text)
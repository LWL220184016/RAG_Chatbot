from ASR.audio_process import Audio_Processer
import queue
import multiprocessing
import nemo.collections.asr as nemo_asr
import traceback


class ASR():
    def __init__(
            self, 
            model: str = "nvidia/parakeet-rnnt-1.1b", 
            device: str = "cuda", 
            ap: Audio_Processer = None, 
            stop_event = None, 
            asr_output_queue: multiprocessing.Queue = None, 
        ):
    
        self.model = nemo_asr.models.ASRModel.from_pretrained(model)
        self.model.eval()
        self.device = device
        self.ap = ap
        self.asr_output_queue = asr_output_queue
        self.stop_event = stop_event

    def asr_output(self, is_asr_ready_event):
        print("asr waiting audio")
        is_asr_ready_event.set()
        while not self.stop_event.is_set():
            # try:
            #     audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            # except queue.Empty:
            #     continue
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = self.ap.process_audio_ws1(audio_data=audio_data)
            try:
                transcriptions = self.model.transcribe(
                    # encoded_features[0],
                    processed_data,
                    batch_size = 4,
                    return_hypotheses = True,
                    verbose = False,
                )
                print("transcriptions: " + str(transcriptions))
                hypothesis = transcriptions[0]
                h = hypothesis[0]
                print("score: " + str(h.score))
                print("You: " + h.text)
                if h.score < 50:
                    continue
                # prompt = input("You: ")
                self.asr_output_queue.put(h.text)
            except Exception as e:
                print("asr_output Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output end")

    def asr_output_ws(self, is_asr_ready_event, asr_output_queue_ws):
        print("asr waiting audio")
        is_asr_ready_event.set()
        while not self.stop_event.is_set():
            # try:
            #     audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            # except queue.Empty:
            #     continue
            audio_data = self.ap.audio_checked_queue.get()
            processed_data = self.ap.process_audio_ws1(audio_data=audio_data)
            try:
                transcriptions = self.model.transcribe(
                    # encoded_features[0],
                    processed_data,
                    batch_size = 4,
                    return_hypotheses = True,
                    verbose = False,
                )
                print("transcriptions: " + str(transcriptions))
                hypothesis = transcriptions[0]
                h = hypothesis[0]
                print("score: " + str(h.score))
                print("You: " + h.text)
                if h.score < 50:
                    continue
                # prompt = input("You: ")
                self.asr_output_queue.put(h.text)
                asr_output_queue_ws.put(h.text)
            except Exception as e:
                print("asr_output Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output end")
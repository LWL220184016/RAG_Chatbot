import queue
import nemo.collections.asr as nemo_asr
import traceback
import logging

from ASR.audio_process import Audio_Processer
from ASR.whisper_streaming.whisper_online import OnlineASRProcessor

class ASR():
    def __init__(
            self, 
            model: str = "nvidia/parakeet-rnnt-1.1b", 
            device: str = "cuda", 
            ap: Audio_Processer = None, 
            stop_event = None, 
            asr_output_queue: queue = None, 
            streaming: bool = False, 
        ):
    
        self.model = nemo_asr.models.ASRModel.from_pretrained(model)
        self.model.eval()
        self.device = device
        self.ap = ap
        self.asr_output_queue = asr_output_queue
        self.stop_event = stop_event

        # only for streaming
        self.transcribe_kargs = {}
        self.logger = logging.getLogger(__name__)
        if streaming:
            print("ASR streaming enabled")
            self.processer = OnlineASRProcessor()
            self.processer.transcribe = self.transcribe
            self.processer.ts_words = self.ts_words
            self.processer.segments_end_ts = self.segments_end_ts
            self.asr_output = self.asr_output_stream
            self.asr_output_ws = self.asr_output_stream_ws

    def asr_output(self, is_asr_ready_event):
        print("asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = self.ap.process_audio2(audio_data=audio_data)
            try:
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
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = self.ap.process_audio2(audio_data=audio_data)
            try:
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
                print("score: " + str(h.score))
                print("You: " + h.text)

                if h.score < 50:
                    continue
                # prompt = input("You: ")
                self.asr_output_queue.put(h.text)
                asr_output_queue_ws.put(h.text)
            except Exception as e:
                print("asr_output_ws Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_ws end")

# only for streaming
    def asr_output_stream(self, is_asr_ready_event):
        import time
        print("asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.processer.insert_audio_chunk(audio_data)
                result = self.processer.process_iter()
                # print("\nASR Output: ", result)
                # print("last ASR text output: ", time.time())
                self.asr_output_queue.put(result[0][2])

            except Exception as e:
                print("asr_output_stream Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_stream end")

    def asr_output_stream_ws(self, is_asr_ready_event, asr_output_queue_ws):
        print("asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.processer.insert_audio_chunk(audio_data)
                result = self.processer.process_iter()
                print("\nASR Output: ", result)
                self.asr_output_queue.put(result[0][2])
                asr_output_queue_ws.put(result[0][2])

            except Exception as e:
                print("asr_output_stream_ws Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_stream_ws end")

    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        # segments = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        # not finish, segments may not suitable and may not output info
        audio = self.ap.process_audio2(audio)
        segments = self.model.transcribe(
            audio,
            batch_size = 4,
            return_hypotheses = True,
            verbose = False,
        )

        return list(segments)

    def ts_words(self, segments):
        o = []
        hypothesis = segments[0]
        h = hypothesis[0]
        t = (1, 2, h.text)
        o.append(t)

        return o

    def segments_end_ts(self, res):

        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

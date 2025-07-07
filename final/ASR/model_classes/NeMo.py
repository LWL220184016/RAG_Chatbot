import queue
import nemo.collections.asr as nemo_asr
import traceback
import logging

from ASR.audio_process import Audio_Processor
from ASR.whisper_streaming.whisper_online import OnlineASRProcessor

class ASR():
    def __init__(
            self, 
            model: str = "nvidia/parakeet-rnnt-1.1b", 
            device: str = "cuda", 
            ap: Audio_Processor = None, 
            stop_event = None, 
            is_user_talking = None, 
            asr_output_queue: queue = None, 
            streaming: bool = False, 
        ):
    
        self.model = nemo_asr.models.ASRModel.from_pretrained(model)
        self.model.eval()
        self.device = device
        self.ap = ap
        self.asr_output_queue = asr_output_queue
        self.stop_event = stop_event
        self.is_user_talking = is_user_talking

        # only for streaming
        self.transcribe_kargs = {}
        self.logger = logging.getLogger(__name__)
        if streaming:
            print("ASR streaming enabled")
            self.processor = OnlineASRProcessor()
            self.processor.transcribe = self.stream_processor_transcribe
            self.processor.ts_words = self.ts_words
            self.processor.segments_end_ts = self.segments_end_ts
            self.asr_output = self.asr_output_stream

    def asr_output(self, is_asr_ready_event, asr_output_queue_ws = None):
        print(f"{self.model} asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed_data = self.ap.process_audio(audio_data=audio_data)

            # Check if audio processing was successful
            if processed_data is None:
                print("Audio processing failed, skipping this chunk")
                continue
            
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
                self.asr_output_queue.put(h.text)
                if not asr_output_queue_ws == None:
                    asr_output_queue_ws.put(h.text)
            except Exception as e:
                print("asr_output Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output end")

# only for streaming
    def asr_output_stream(self, 
                          is_asr_ready_event, 
                          asr_output_queue_ws = None, 
                          clean_buffer_timeout=5
                         ):
        print(f"{self.model} asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.processor.insert_audio_chunk(audio_data, clean_buffer_timeout)
                result = self.processor.process_iter()
                print("\nASR Output: ", result)
                if not self.is_user_talking.is_set() and self.ap.audio_checked_queue.empty():
                    self.asr_output_queue.put(result[0][2])
                    if not asr_output_queue_ws == None:
                        asr_output_queue_ws.put(result[0][2])

            except Exception as e:
                print("asr_output_stream Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_stream end")

    def stream_processor_transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        # segments = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        # not finish, segments may not suitable and may not output info
        audio = self.ap.process_audio(audio)
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
        print(res)
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"



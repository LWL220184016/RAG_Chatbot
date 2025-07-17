import queue
import traceback
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging

from ASR.audio_process import Audio_Processor
from ASR.whisper_streaming.whisper_online import OnlineASRProcessor

class ASR():
    def __init__(
            self, 
            model: str = "openai/whisper-medium", # "openai/whisper-large-v3", "openai/whisper-medium"
            device: str = "cuda", 
            ap: Audio_Processor = None, 
            stop_event = None, 
            is_user_talking = None, 
            asr_output_queue: queue = None, 
            streaming: bool = False, 
        ):
    
        self.asr_processor = WhisperProcessor.from_pretrained(model, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, local_files_only=True).to(device)
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

            try:
                processed_data = self.ap.process_audio(audio_data=audio_data)
                
                # Check if audio processing was successful
                if processed_data is None:
                    print("Audio processing failed, skipping this chunk")
                    continue
                
                input_features = self.asr_processor(processed_data, sampling_rate=16000, return_tensors="pt").input_features

                predicted_ids = self.model.generate(input_features.to(self.device))[0]
                transcriptions = self.asr_processor.decode(predicted_ids)
                print("transcriptions: " + transcriptions)

                # 尚未實作（計算分數，如果過低則不輸出）

                self.asr_output_queue.put(transcriptions)
                print("asr_output_queue size: ", self.asr_output_queue.qsize())
                if not asr_output_queue_ws == None:
                    # asr_output_queue_ws.put(h.text)
                    asr_output_queue_ws.put(transcriptions)
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
                print("processing audio--------------")
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
        audio = self.ap.process_audio(audio)
        input_features = self.asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features

        predicted_ids = self.model.generate(input_features.to(self.device))[0]
        transcriptions = self.asr_processor.decode(predicted_ids, skip_special_tokens=True)
        print("transcriptions: ", transcriptions)
        return transcriptions

    def ts_words(self, segments):
        o = []
        t = (1, 2, segments)
        o.append(t)

        return o
    
    # todo
    # # from whisper_streaming, for reference when throw error
    # def ts_words(self, segments):
    #     o = []
    #     print(segments)
    #     for segment in segments:
    #         for word in segment.words:
    #             if segment.no_speech_prob > 0.8:
    #                 continue
    #             # not stripping the spaces -- should not be merged with them!
    #             w = word.word
    #             t = (word.start, word.end, w)
    #             o.append(t)
    #             print("w: ", w)
    # # [Segment(id=1, seek=0, start=0.0, end=0.36, text=' Hello?', tokens=[50364, 2425, 30, 50389], avg_logprob=-0.6265625, compression_ratio=0.42857142857142855, no_speech_prob=0.03387451171875, words=[Word(start=0.0, end=0.36, word=' Hello?', probability=0.80126953125)], temperature=0.0)]

    #         return o


    def segments_end_ts(self, res):
        print(res)
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


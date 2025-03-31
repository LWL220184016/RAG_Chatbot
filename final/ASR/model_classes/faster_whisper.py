import queue
import traceback
import logging

from faster_whisper import WhisperModel
from ASR.audio_process import Audio_Processer
from ASR.whisper_streaming.whisper_online import OnlineASRProcessor

class ASR():
    def __init__(
            self, 
            model: str = "large-v3", 
            device: str = "cuda", 
            compute_type: str = "float16", 
            ap: Audio_Processer = None, 
            stop_event = None, 
            is_user_talking = None, 
            asr_output_queue: queue = None, 
            streaming: bool = False, 
        ):
    
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
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
                segments, info = self.model.transcribe(
                    # encoded_features[0],
                    processed_data, 
                    beam_size=5, 
                ) 
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
                segments, info = self.model.transcribe(
                    # encoded_features[0],
                    processed_data, 
                    beam_size=5, 
                ) 
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
                asr_output_queue_ws.put(prompt)
            except Exception as e:
                print("asr_output_ws Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_ws end")

the following code just save with NeMo may not match to faster whisper, 
need to modify the following code to match faster_whisper
# only for streaming
    def asr_output_stream(self, 
                          is_asr_ready_event, 
                          user_talk_timeout=0.2, 
                          clean_buffer_timeout=5
                        ):
        """
        user_talk_timeout: float = 0.2, # 如果用戶在 0.2 秒內沒有說話，則認爲用戶已經停止說話
        """
        import time
        print("asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.processer.insert_audio_chunk(audio_data, clean_buffer_timeout)
                result = self.processer.process_iter()
                # print("\nASR Output: ", result)
                # print("last ASR text output: ", time.time())
                if not self.is_user_talking.is_set() and self.ap.audio_checked_queue.empty():
                    self.asr_output_queue.put(result[0][2])

            except Exception as e:
                print("asr_output_stream Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_stream end")

    def asr_output_stream_ws(self, 
                             is_asr_ready_event, 
                             asr_output_queue_ws, 
                             user_talk_timeout=0.2, 
                             clean_buffer_timeout=5
                            ):
        print("asr waiting audio")
        is_asr_ready_event.set()

        while not self.stop_event.is_set():
            try:
                audio_data = self.ap.audio_checked_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.processer.insert_audio_chunk(audio_data, clean_buffer_timeout)
                result = self.processer.process_iter()
                print("\nASR Output: ", result)
                if not self.is_user_talking.is_set() and self.ap.audio_checked_queue.empty():
                    self.asr_output_queue.put(result[0][2])
                    asr_output_queue_ws.put(result[0][2])

            except Exception as e:
                print("asr_output_stream_ws Exception: " + str(e))
                traceback.print_exc()
                continue
        print("asr_output_stream_ws end")

    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)

        #print(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments):
        o = []
        print(segments)
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)

# [Segment(id=1, seek=0, start=0.0, end=0.36, text=' Hello?', tokens=[50364, 2425, 30, 50389], avg_logprob=-0.6265625, compression_ratio=0.42857142857142855, no_speech_prob=0.03387451171875, words=[Word(start=0.0, end=0.36, word=' Hello?', probability=0.80126953125)], temperature=0.0)]

        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

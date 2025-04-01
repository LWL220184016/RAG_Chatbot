class ASR:
    def __init__(self, model_name, device, language, chunk_size=4096):
        pass

    def asr_output(self, is_asr_ready_event):
        pass

    def asr_output_stream(self, 
                        is_asr_ready_event, 
                        user_talk_timeout=0.2, 
                        clean_buffer_timeout=5
                    ):
        pass

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("Subclasses should implement this method")

    def ts_words(self, segments):
        raise NotImplementedError("Subclasses should implement this method")

    def segments_end_ts(self, res):
        raise NotImplementedError("Subclasses should implement this method")

    def use_vad(self):
        raise NotImplementedError("Subclasses should implement this method")

    def set_translate_task(self):
        raise NotImplementedError("Subclasses should implement this method")

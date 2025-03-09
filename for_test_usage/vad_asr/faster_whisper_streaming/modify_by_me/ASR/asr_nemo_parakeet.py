import logging
from ASR.asr_base import ASRBase

class NeMoParakeetASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """
    def __init__(self, modelsize=None, cache_dir=None, model_dir=None, model_name="nvidia/parakeet-rnnt-1.1b", original_language="en"):
        super().__init__()
        self.model = self.load_model(model_name)
        self.original_language = original_language
        self.transcribe_kargs = {}
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_name="nvidia/parakeet-rnnt-1.1b", modelsize=None, cache_dir=None, model_dir=None):
        import nemo.collections.asr as nemo_asr
        
        try: 
            import torch
            if torch.cuda.is_available():
                print("CUDA is available, running on GPU")
                device = "cuda"
                # this worked fast and reliably on NVIDIA L40
                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=device)
                model.eval()

                # or run on GPU with INT8
                # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
                #model = WhisperModel(model_size, device=device, compute_type="int8_float16")

            else:
                print("CUDA is not available, running on CPU")
                device = "cpu"
                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=device)
                model.eval()

        except Exception:
            device = "cpu"
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=device)
            model.eval()

        return model

    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        # segments = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        # not finish, segments may not suitable and may not output info
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
        print("h: ", h)
        t = (1, 2, h.text)
        o.append(t)

        # o = []
        # for segment in segments:
        #     print(segment)
        #     for word in segment.text:
        #         if segment.no_speech_prob > 0.9:
        #             continue
        #         # not stripping the spaces -- should not be merged with them!
        #         w = word.word
        #         t = (word.start, word.end, w)
        #         o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

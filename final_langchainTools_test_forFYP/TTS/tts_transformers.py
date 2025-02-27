import multiprocessing
import queue
import time
import torch

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

class TTS():
    def __init__(
            self, 
            processor: str = "microsoft/speecht5_tts",
            model: str = "microsoft/speecht5_tts",
            vocoder: str = "microsoft/speecht5_hifigan",
            embeddings_dataset: str = "Matthijs/cmu-arctic-xvectors",
            embeddings_split: str = "validation",
            speaker_embeddings: str = None,
            stop_event = None,
            audio_queue: multiprocessing.Queue = None,
        ):

        self.processor = SpeechT5Processor.from_pretrained(processor) 
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model) 
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder)
        self.embeddings_dataset = load_dataset(embeddings_dataset, split=embeddings_split) 
        self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0) 
        self.audio_queue = audio_queue
        self.stop_event = stop_event

    def tts_output(self, speaking_event, is_tts_ready_event, llm_output_queue: queue.Queue):
        print("tts waiting text")
        is_tts_ready_event.set()
        while not self.stop_event.is_set():
            if llm_output_queue.empty():
                speaking_event.clear()  # Signal that LLM has finished speaking
            try:
                text = llm_output_queue.get(timeout=1)
            except queue.Empty:
                continue
            while self.audio_queue.qsize() >= 5:
                time.sleep(0.01)
                pass
            inputs = self.processor(text=text, return_tensors="pt") 
            audio_chunk = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            audio_chunk = audio_chunk.cpu().numpy()
            self.audio_queue.put(audio_chunk)
            speaking_event.set()  # Signal that LLM is speaking
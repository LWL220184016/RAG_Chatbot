import threading
import queue
import time
from langchain_ollama import OllamaLLM
import base64
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset 
import torch 
import sounddevice as sd 
import numpy as np

def get_LLM():
    # Initialize langchain ollama with GGUF format model
    langchain_llm = OllamaLLM(
        # model="Qwen2-VL-7B-Instruct",
        # model="llama3.2-vision",
        # model="llama3.3:70b-instruct-q2_K",
        model="llama3.2-vision-latest-friend2",
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    return langchain_llm

def get_TTS():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") 
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts") 
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan") 

    # Load speaker embeddings 
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") 
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) 
    return processor, model, vocoder, speaker_embeddings


def llm_output(llm, user_input_queue, llm_output_queue):
    while True:
        prompt = user_input_queue.get()
        text = ""
        for output in llm.invoke(prompt):
            if output not in ["，", ",", "。", ".", "？", "?", "！", "!"]:
                text += output
            else:
                llm_output_queue.put(text)
                text = ""
        user_input_queue.task_done()

def tts_output(processor, model, vocoder, speaker_embeddings, llm_output_queue, audio_queue, speaking_event):
    while True:
        if llm_output_queue.empty():
            speaking_event.clear()  # Signal that LLM has finished speaking

        text = llm_output_queue.get()
        while audio_queue.qsize() >= 5:
            pass
        print("LLM: ", text)
        inputs = processor(text=text, return_tensors="pt") 
        audio_chunk = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        audio_chunk = audio_chunk.cpu().numpy()
        audio_queue.put(audio_chunk)
        speaking_event.set()  # Signal that LLM is speaking
        llm_output_queue.task_done()



if __name__ == "__main__":
    # Load the processor, model, and vocoder 
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") 
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts") 
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan") 

    # Load speaker embeddings 
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") 
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) 

    llm = get_LLM()

    user_input_queue = queue.Queue()
    llm_output_queue = queue.Queue()
    audio_queue = queue.Queue()
    speaking_event = threading.Event()

    llm_thread = threading.Thread(target=llm_output, args=(llm, user_input_queue, llm_output_queue))
    tts_thread = threading.Thread(target=tts_output, args=(processor, model, vocoder, speaker_embeddings, llm_output_queue, audio_queue, speaking_event))

    llm_thread.start()
    tts_thread.start()

    # for audio_data in record_and_detect_vad():

    try:
        while True:
            prompt = input("You: ")
            user_input_queue.put(prompt)
            speaking_event.set()  # Signal that LLM is speaking
            
            # while not audio_queue.empty():
            while speaking_event.is_set() or not audio_queue.empty():
                audio_chunk = audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=True)
                sd.wait()
                audio_queue.task_done()

    except KeyboardInterrupt:
        pass
    finally:
        user_input_queue.put(None)
        llm_output_queue.put(None)
        llm_thread.join()
        tts_thread.join()


# pip install transformers datasets torch numpy sentencepiece sounddevice 
# sudo apt-get install portaudio19-dev
# sudo apt-get update
# sudo apt-get install pulseaudio

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset 
import torch 
import sounddevice as sd 
import time

import threading
import queue

def tts_output(processor, model, vocoder, output_queue):
    Process_time = time.time()
    for i in range(output_queue.qsize()):
        text = output_queue.get()
        print("text: ", text)
        if text is None:
            break
        inputs = processor(text=text, return_tensors="pt").to(device) 
        audio_chunk = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        audio_chunk = audio_chunk.cpu().numpy()
        audio_queue.put(audio_chunk)
        print("Process_time: ", time.time()-Process_time)
        
        output_queue.task_done()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the processor, model, and vocoder 

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") 
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device) 
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device) 

# Load speaker embeddings 
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") 
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device) 

output_queue = queue.Queue()
audio_queue = queue.Queue()

# Input text 
text = "what is your name?" 
output_queue.put(text)
text = "what is your quest?" 
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 1
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 2
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 3
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 4
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 5
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 6
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 7
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 8
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 9
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 1
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 2
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 3
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 4
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 5
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 6
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 7
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 8
output_queue.put(text)
text = "what is the airspeed velocity of an unladen swallow?" # 9
output_queue.put(text)

Total_time = time.time()

process_tts = threading.Thread(target=tts_output, args=(processor, model, vocoder, output_queue))
process_tts.start()

for i in range(output_queue.qsize()):
    audio_chunk = audio_queue.get()

    # Time2 = time.time()
    # print("Time1: ", time.time()-Time1, "   Time2: ", time.time()-Time2)
    sd.play(audio_chunk, samplerate=16000) 
    sd.wait()
    print("Total: ", time.time()-Total_time)

# pip install transformers datasets torch numpy sentencepiece sounddevice 
# sudo apt-get install portaudio19-dev
# sudo apt-get update
# sudo apt-get install pulseaudio

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset 
import torch 
import sounddevice as sd 
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the processor, model, and vocoder 
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") 
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device) 
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device) 

# Load speaker embeddings 
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") 
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device) 

# Input text 
text = "what is your name? what is your quest? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow? what is the airspeed velocity of an unladen swallow?" 
Process_time = time.time()
inputs = processor(text=text, return_tensors="pt").to(device) 
audio_chunk = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# Check the shape and dtype of audio_chunk
audio_chunk = audio_chunk.cpu().numpy()
print("Audio chunk shape:", audio_chunk.shape)
print("Audio chunk dtype:", audio_chunk.dtype)

print("Process time: ", time.time()-Process_time)

Audio_play_time = time.time()
sd.play(audio_chunk, samplerate=16000) 
sd.wait()
print("Audio play time: ", time.time()-Audio_play_time)
print("total: ", time.time()-Process_time)
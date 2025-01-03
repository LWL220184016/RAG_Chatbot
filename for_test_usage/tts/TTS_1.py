# pip install transformers datasets torch numpy sentencepiece sounddevice 
# sudo apt-get install portaudio19-dev
# sudo apt-get update
# sudo apt-get install pulseaudio

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset 
import torch 
import sounddevice as sd 
import time
Time1 = time.time()

    # Load the processor, model, and vocoder 
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") 
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts") 
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan") 

# Load speaker embeddings 
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") 
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) 

# Input text 
text = "what is your name? what is your quest? what is the airspeed velocity of an unladen swallow?" 
Time2 = time.time()

inputs = processor(text=text, return_tensors="pt") 

# Generate spectrogram in chunks 
chunk_size = 500

# Number of spectrogram frames per chunk 
spectrogram_chunks = [] 

# Generate spectrogram in chunks 
for i in range(0, inputs["input_ids"].shape[1], chunk_size): 
    chunk = inputs["input_ids"][:, i:i + chunk_size] 

spectrogram_chunk = model.generate_speech(chunk, speaker_embeddings) 
spectrogram_chunks.append(spectrogram_chunk) 

# Concatenate spectrogram chunks 
spectrogram = torch.cat(spectrogram_chunks, dim=0) 

# Convert spectrogram to audio in chunks and stream it 
audio_chunks = [] 

for i in range(0, spectrogram.shape[0], chunk_size): 
    spectrogram_chunk = spectrogram[i:i + chunk_size].unsqueeze(0) 
    audio_chunk = vocoder(spectrogram_chunk).detach().numpy() 
    audio_chunks.append(audio_chunk) 
    
    # Play the audio chunk as it is generated 
    audio_chunk = audio_chunk.squeeze() 
    # Reshape (1, 25600) to (25600,) 
    print("Time1: ", time.time()-Time1, "   Time2: ", time.time()-Time2)
    sd.play(audio_chunk, samplerate=16000) 
    sd.wait()
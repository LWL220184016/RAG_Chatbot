from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from datasets import load_dataset
import torchaudio
import numpy as np  

def get_asr_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "distil-whisper/distil-large-v3"
    model_id = "distil-whisper/distil-medium.en"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, device, torch_dtype

if __name__ == "__main__":
    model, processor, device, torch_dtype = get_asr_model()
    
    audio_file_path = "en_example.wav"
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # Convert to mono if the audio has more than one channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Ensure the sample rate matches the model's expected sample rate
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
        sample_rate = 16000

    # Convert waveform to the format expected by the pipeline
    sample = {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}

    input_features = processor(sample["array"], sampling_rate=sample_rate, return_tensors="pt")
    input_features = input_features.to(device, dtype=torch_dtype)

    pred_ids = model.generate(input_features.input_features)
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

    print("\naudio file: " + pred_text)

from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from datasets import load_dataset
import torchaudio
import numpy as np  
import pyaudio
import sys
import threading
import queue

CHUNK = 512  # Reduce chunk size to avoid overflow
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
有点问题, 能处理一点但是效果不太好, 怀疑是硬件性能导致的
p = pyaudio.PyAudio()
stereo_mix_index = None
print(p.get_device_count())

for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if "Stereo Mix" in dev_info.get('name', ''):
        stereo_mix_index = i
        break

if stereo_mix_index is None:
    print("Stereo Mix device not found")
    sys.exit(1)

def get_audio_chunk(stream, audio_chunk_queue, stop_event):
    try:
        while stop_event.is_set() == False:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk_queue.put(frame)
    except Exception as e:
        print(e.message)

def convert_audio_bytes_to_float(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def normalize_audio(audio_np):
    return (audio_np - np.mean(audio_np)) / np.std(audio_np)

def get_asr_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "distil-whisper/distil-large-v3"
    model_id = "distil-whisper/distil-medium.en"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, device, torch_dtype

if __name__ == "__main__":
    fileNum = 0
    audio_chunk_queue = queue.Queue(50)
    stop_event = threading.Event()
    audio_buffer = []

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=stereo_mix_index, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk, args=(stream, audio_chunk_queue, stop_event))
        get_audio_threading.start()
        asr_model, asr_processor, device, torch_dtype = get_asr_model()
        while True:
            frame = audio_chunk_queue.get()
            audio_np = convert_audio_bytes_to_float(frame)
            audio_buffer.extend(audio_np)

            if len(audio_buffer) >= RATE * 5:  # Process every 5 seconds of audio
                audio_data = np.array(audio_buffer[:RATE * 5])
                audio_buffer = audio_buffer[RATE * 5:]

                audio_data = normalize_audio(audio_data)
                input_features = asr_processor(
                    audio_data, sampling_rate=RATE, return_tensors="pt"
                )
                input_features = input_features.to(device, dtype=torch_dtype)
                # print(len(frame))

                pred_ids = asr_model.generate(input_features.input_features)
                pred_text = asr_processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

                print("\naudio file: " + pred_text)
    except KeyboardInterrupt:
        stop_event.set()
        get_audio_threading.join()
        stream.stop_stream()
        stream.close()
        p.terminate()
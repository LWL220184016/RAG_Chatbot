from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from datasets import load_dataset
import numpy as np  
import pyaudio
import threading
import queue
import noisereduce as nr

SOUND_LEVEL = 100
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SEC = 1

def convert_audio_bytes_to_float(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def normalize_audio(audio_np):
    return (audio_np - np.mean(audio_np)) / np.std(audio_np)

def get_asr_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "distil-whisper/distil-large-v3"
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, device, torch_dtype

def reduce_noise(audio_np, rate):
    reduced_noise = nr.reduce_noise(y=audio_np, sr=rate)
    return reduced_noise

if __name__ == "__main__":
    from check_input_device import get_input_device, get_audio_chunk, get_audio_chunk_sound_level
    
    p = pyaudio.PyAudio()
    # sound_device_index = get_input_device(p, "Microphone (MONSTER AIRMARS N3)")

    fileNum = 0
    audio_chunk_queue = queue.Queue(50)
    stop_event = threading.Event()

    try:
        # stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=sound_device_index, frames_per_buffer=CHUNK)
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk_sound_level, args=(stream, audio_chunk_queue, stop_event, CHUNK, SOUND_LEVEL))
        # get_audio_threading = threading.Thread(target=get_audio_chunk, args=(stream, audio_chunk_queue, stop_event, CHUNK))
        get_audio_threading.start()
        asr_model, asr_processor, device, torch_dtype = get_asr_model()
        print("Recording...")
        while True:

            # frame = audio_chunk_queue.get()
            # audio_data = convert_audio_bytes_to_float(frame)

            # # audio_data = normalize_audio(audio_data)
            # input_features = asr_processor(
            #     audio_data, sampling_rate=RATE, return_tensors="pt"
            # )
            # input_features = input_features.to(device, dtype=torch_dtype)


            frame = audio_chunk_queue.get()
            audio_data = convert_audio_bytes_to_float(frame)

            # Reduce noise
            audio_data = reduce_noise(audio_data, RATE)

            # Normalize audio
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
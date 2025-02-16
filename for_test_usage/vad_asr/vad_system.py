import pyaudio
import torch
import numpy as np
from silero_vad import get_speech_timestamps, read_audio#, VADIterator
from vad import VADIterator
import time
import sys
import wave
import threading
import queue
from check_input_device import get_input_device, get_audio_chunk

CHUNK = 512  # Reduce chunk size to avoid overflow
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def convert_audio_bytes_to_float(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_np = int2float(audio_np)
    return audio_np

def record_and_detect_vad(audio_chunk_queue, stream, vad):
    frames = []
    try:
        while True:
            frames = audio_chunk_queue.get()
            _frame = frames
            if frames is not None:
                frames = convert_audio_bytes_to_float(frames)
                frames = np.copy(frames)  # Make the array writable
                sound_frame = vad(frames, _frame)
                if sound_frame is not None:
                    vad.reset_states()
                    yield sound_frame
                    
    except OSError as e:
        print(f"Error reading stream: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    p = pyaudio.PyAudio()
    sound_device_index = get_input_device(p, "Microphone (MONSTER AIRMARS N3)")

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    threshold = 0.3
    sampling_rate = 16000
    min_silence_duration_ms = 1000
    speech_pad_ms=30
    vad = VADIterator(model, 
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
    )
    
    fileNum = 0
    audio_chunk_queue = queue.Queue(50)
    stop_event = threading.Event()

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=sound_device_index, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk, args=(stream, audio_chunk_queue, stop_event, CHUNK))
        get_audio_threading.start()

        for frame in record_and_detect_vad(audio_chunk_queue, stream, vad):
            print(len(frame))
            fileNum += 1
            with wave.open('sound_files/output' + str(fileNum) + '.wav', 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)

                print('Recording file ' + str(fileNum) + '...')
                for i in range(0, len(frame)):
                    wf.writeframes(frame[i])
                print('Done')
    except KeyboardInterrupt:
        print("Recording stopped by user")
    finally:
        stop_event.set()
        if stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()



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

CHUNK = 512  # Reduce chunk size to avoid overflow
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

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

def record_and_detect_vad(audio_chunk_queue, stream):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    vad = VADIterator(model)
    sample_rate = 16000
    frame_duration = 30  # ms
    padding_duration = 300  # ms

    frames = []

    try:
        while True:
            frames = audio_chunk_queue.get()
            _frame = frames
            if frames is not None:
                frames = np.frombuffer(frames, dtype=np.int16)
                frames = np.copy(frames)  # Make the array writable
                sound_frame = vad(frames, _frame)
                if sound_frame is not None:
                    yield sound_frame
    except OSError as e:
        print(f"Error reading stream: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    fileNum = 0
    audio_chunk_queue = queue.Queue(50)
    stop_event = threading.Event()

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=stereo_mix_index, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk, args=(stream, audio_chunk_queue, stop_event))
        get_audio_threading.start()
        for frame in record_and_detect_vad(audio_chunk_queue, stream):
            print(len(frame))
            fileNum += 1
            with wave.open('sound_files/output' + str(fileNum) + '.wav', 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)

                print('Recording...')
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



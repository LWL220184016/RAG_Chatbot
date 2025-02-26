import pyaudio
import sys
import numpy as np
import time

# input_device_name = "CABLE Output"

def get_input_device(p, input_device_name):
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if input_device_name in dev_info.get('name', ''):
            sound_device_index = i
            break

    if sound_device_index is None:
        print("CABLE Output device not found")
        sys.exit(1)

def show_input_device(p):
    print(p.get_device_count())
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print("index: " + str(i) + " ; " + dev_info.get('name', ''))

def get_audio_chunk(stream, audio_chunk_queue, stop_event, CHUNK):
    while not stop_event.is_set():
        try:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk_queue.put(frame)
        except OSError as e:
            print(f"OSError: {e}")

def get_audio_chunk_sound_level(stream, audio_chunk_queue, stop_event, CHUNK, sound_level_threshold=100, timeout_sec=0.5):
    frames = bytearray()
    record_start_time = 0
    while not stop_event.is_set(): # 加入一個參數輸入chunk 數量或者毫秒，當聲音強度低過閾值時，等待一段時間，再檢查聲音強度
        try:
            frame = stream.read(CHUNK)
            audio_data = np.frombuffer(frame, dtype=np.int16)
            # 計算聲音強度
            volume_norm = np.linalg.norm(audio_data) / CHUNK
            
            if volume_norm > sound_level_threshold:
                # print("Sound detected, ", f'聲音強度: {volume_norm:.2f}')
                frames.extend(frame)
                record_start_time = time.time()

            else:
                if len(frames) > 0:
                    if time.time() - record_start_time < timeout_sec:
                        frames.extend(frame)
                        continue
                    audio_chunk_queue.put(bytes(frames))
                    frames = bytearray()
                    record_start_time = 0
                    print(f'聲音強度: {volume_norm:.2f}')
        except OSError as e:
            print(f"OSError: {e}")

if __name__ == '__main__':
    p = pyaudio.PyAudio()
    show_input_device(p)
    # get_input_device(p, input_device_name)
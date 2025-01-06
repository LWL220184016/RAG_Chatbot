import pyaudio
import queue
import numpy as np
import time
import threading
import noisereduce as nr

class Audio_Processer():
    def __init__(
            self, 
            chunk = 512,
            format = pyaudio.paInt16,
            channels = 1,
            rate = 16000,
            sec = 1,
            audio_unchecked_queue = queue.Queue(),
            audio_checked_queue = queue.Queue()
        ):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.sec = sec
        self.audio_unchecked_queue = audio_unchecked_queue
        self.audio_checked_queue = audio_checked_queue
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        self.stop_event = threading.Event()
        

    def get_chunk(self, exception_on_overflow):
        while not self.stop_event.is_set():
            self.audio_unchecked_queue.put(self.stream.read(self.chunk, exception_on_overflow=exception_on_overflow))
    
    def detect_sound(self, sound_level_threshold = 100, timeout_sec=0.5):
        frames = bytearray()
        record_start_time = 0
        while not self.stop_event.is_set(): # 加入一個參數輸入chunk 數量或者毫秒，當聲音強度低過閾值時，等待一段時間，再檢查聲音強度
            try:
                frame = self.audio_unchecked_queue.get()
                audio_data = np.frombuffer(frame, dtype=np.int16)
                # 計算聲音強度
                volume_norm = np.linalg.norm(audio_data) / self.chunk
                
                if volume_norm > sound_level_threshold:
                    # print("Sound detected, ", f'聲音強度: {volume_norm:.2f}')
                    frames.extend(frame)
                    record_start_time = time.time()

                else:
                    if len(frames) > 0:
                        if time.time() - record_start_time < timeout_sec:
                            frames.extend(frame)
                            continue
                        self.audio_checked_queue.put(bytes(frames))
                        frames = bytearray()
                        record_start_time = 0
                        print(f'聲音強度: {volume_norm:.2f}')
            except OSError as e:
                print(f"OSError: {e}")

    def process_audio1(self, audio_data, asr_processor, device, torch_dtype):
        audio_data = np.frombuffer(audio_data, dtype = np.int16).astype(np.float32) / 32768.0 # audio bytes to float
        # audio_data = normalize_audio(audio_data)
        audio_data = asr_processor(
            audio_data, sampling_rate=self.rate, return_tensors="pt"
        )
        audio_data = audio_data.to(device, dtype=torch_dtype)

        
    def process_audio2(self, audio_data):
        audio_data = np.frombuffer(audio_data, dtype = np.int16).astype(np.float32) / 32768.0 # audio bytes to float
        audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data) # normalize audio float data
        audio_data = nr.reduce_noise(y = audio_data, sr = self.rate) # reduced noise
        return audio_data
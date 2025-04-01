import queue
import numpy as np
import time
import noisereduce as nr
import torchaudio
import torch

from pydub import AudioSegment
import io

class Audio_Processor():
    """
    startStream: bool
        If True, start stream when init. 
        Should not use in server client approach and Web Socket, 
        because the stream is use to get audio data from microphone. 

        when using server client approach, the audio data should 
        be sent from client to server.

    is_user_talking: multiprocessing.Event
        For stop llm invoking when user is talking

    stop_event: multiprocessing.Event 
        For stop all threads and multiprocessing
    """
    def __init__(
            self, 
            sound_process_library = "sounddevice", # Choose a library, "pyaudio" or "sounddevice"
            chunk: int = 512, 
            format = np.float32, # default, auto change to match sound_process_library
            channels: int = 1, 
            rate: int = 16000, 
            audio_unchecked_queue: queue = queue.Queue(), 
            audio_checked_queue: queue = queue.Queue(), 
            startStream: bool = True,
            is_user_talking = None,
            stop_event = None,
            input_device_index: int = 1, 
        ):
        self.rate = rate
        self.chunk = chunk
        self.audio_unchecked_queue = audio_unchecked_queue
        self.audio_checked_queue = audio_checked_queue
        if startStream:
            self._setup_sound_process_library(sound_process_library, channels, input_device_index)

        self.is_user_talking = is_user_talking
        self.stop_event = stop_event
        self.mel_spectrogram = None

        # todo
        # pyaudio float32 test ok, need to test sounddevice float 32
        # test time different in pyaudio float32 and pyaudio int16
    def _setup_sound_process_library(self, 
                                     sound_process_library, 
                                     channels, 
                                     input_device_index, 
                                    ):
        if sound_process_library == "pyaudio":
            import pyaudio

            self.stream = pyaudio.PyAudio().open(
                format=pyaudio.paFloat32, 
                channels=channels, 
                rate=self.rate, 
                input=True, 
                frames_per_buffer=self.chunk, 
                input_device_index=input_device_index
            )
        
        elif sound_process_library == "sounddevice":
            import sounddevice

            print("Loading Audio Processor with sounddevice")
            self.stream = sounddevice.InputStream(  # Use OutputStream for playback
                samplerate=self.rate, 
                blocksize=self.chunk, 
                channels=channels, 
                dtype=np.float32, 
                device=input_device_index, # Use device index or name
            )
            self.stream.start()  # Start the stream explicitly
            self.get_chunk = self.get_chunk_sd

    def set_mel_spectrogram(
            self, 
            rate: int = None, 
            n_mels: int = 80, 
            n_fft: int = 400, 
            hop_length: int = 160
        ):

        if rate is None:
            rate = self.rate

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def get_chunk(self, is_asr_ready_event):
        while not is_asr_ready_event.is_set():
            time.sleep(0.1)
        while not self.stop_event.is_set():
            self.audio_unchecked_queue.put(self.stream.read(self.chunk, exception_on_overflow=True))
    # use when the sound_process_library is sounddevice
    def get_chunk_sd(self, is_asr_ready_event):
        while not is_asr_ready_event.is_set():
            time.sleep(0.1)
        while not self.stop_event.is_set():
            data = self.stream.read(self.chunk)
            self.audio_unchecked_queue.put(data[0])

    def detect_sound(
            self, 
            sound_level_threshold: int = 100, 
            timeout_sec: float = 0.5
        ):

        frames = bytearray()
        record_start_time = 0
        while not self.stop_event.is_set(): # 加入一個參數輸入chunk 數量或者毫秒，當聲音強度低過閾值時，等待一段時間，再檢查聲音強度
            try:
                try:
                    frame = self.audio_unchecked_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                audio_data = np.frombuffer(frame, dtype=np.float32)
                # 計算聲音強度
                
                # for dtype is np.int16
                # sound_volume = np.linalg.norm(audio_data) / self.chunk

                # for dtype is np.float32
                sound_volume = np.sqrt(np.mean(audio_data**2))
                sound_volume *= 40 # After calculation, the value is only a few tenths, which is too small.

                if sound_volume > sound_level_threshold:
                    # print("\nRecording...")
                    self.is_user_talking.set()
                    # print("Sound detected, ", f'聲音強度: {sound_volume:.2f}')
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
                        self.is_user_talking.clear()
                        # print(f'聲音強度: {sound_volume:.2f}')

            except Exception as e:
                import traceback
                traceback.print_exc()
    
    def detect_sound_not_extend(
            self, 
            sound_level_threshold: int = 100, 
            timeout_sec: float = 0.5
        ):

        # frames = bytearray()
        record_start_time = 0
        while not self.stop_event.is_set(): # 加入一個參數輸入chunk 數量或者毫秒，當聲音強度低過閾值時，等待一段時間，再檢查聲音強度
            try:
                frame = self.audio_unchecked_queue.get()
                audio_data = np.frombuffer(frame, dtype=np.float32)
                # 計算聲音強度

                # for dtype is np.int16
                # sound_volume = np.linalg.norm(audio_data) / self.chunk

                # for dtype is np.float32
                sound_volume = np.sqrt(np.mean(audio_data**2))
                sound_volume *= 40 # After calculation, the value is only a few tenths, which is too small.

                if sound_volume > sound_level_threshold:
                    # print("Sound detected, ", f'聲音強度: {sound_volume:.2f}')
                    # print("last sound detect: ", time.time())
                    # self.audio_checked_queue.put(bytes(frames))
                    self.audio_checked_queue.put(frame)
                    record_start_time = time.time()
                    self.is_user_talking.set()

                else:
                    # if len(frames) > 0:
                        if time.time() - record_start_time < timeout_sec:
                            # self.audio_checked_queue.put(bytes(frames))
                            self.audio_checked_queue.put(frame)
                            continue
                        # frames = bytearray()
                        record_start_time = 0
                        self.is_user_talking.clear()
                        # print(f'聲音強度: {sound_volume:.2f}')
            except OSError as e:
                import traceback
                traceback.print_exc()

    def process_audio1(self, audio_data, asr_processor, device, torch_dtype) -> None:
        # for int16
        # audio_data = np.frombuffer(audio_data, dtype = np.int16).astype(np.float32) / 32768.0 # audio bytes to float

        # for float32
        audio_data = np.frombuffer(audio_data, dtype = np.float32) / 32768.0


        # audio_data = normalize_audio(audio_data)
        audio_data = asr_processor(
            audio_data, sampling_rate=self.rate, return_tensors="pt"
        )
        audio_data = audio_data.to(device, dtype=torch_dtype)

    def process_audio2(
            self, 
            audio_data: bytes,
        ) -> np.ndarray:

        try:
            # for int16
            # audio_data = np.frombuffer(audio_data, dtype = np.int16).astype(np.float32) / 32768.0 # audio bytes to float

            # for float32
            audio_data = np.frombuffer(audio_data, dtype = np.float32) / 32768.0

            audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data) # normalize audio float data
            audio_data = nr.reduce_noise(y = audio_data, sr = self.rate) # reduced noise
        
            return audio_data
        except ValueError as e:
            print(f"ValueError: {e}")
            return None

    def process_audio_ws1(
            self, 
            audio_data: bytes,
        ) -> np.ndarray:
        """
        this function have a problem, may return None type data
        """

        try:
            # Convert to AudioSegment object
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            
            # Unify the sampling rate
            audio = audio.set_frame_rate(16000)
            
            # Convert to mono(单声道)
            audio = audio.set_channels(1)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples = np.iinfo(audio.array_type).max  # Normalized to [-1, 1]
            
            return samples
        except Exception as e:
            print(f"Processing Error: {e}")
            return None

    def process_audio3(
            self, 
            audio_data: bytes,
        ) -> torch.Tensor:
        # for int16
        # audio_data = np.frombuffer(audio_data, dtype = np.int16).astype(np.float32) / 32768.0 # audio bytes to float

        # for float32
        audio_data = np.frombuffer(audio_data, dtype = np.float32) / 32768.0

        mel = self.mel_spectrogram(audio_data)
        mel = mel.unsqueeze(0)  # Shape: (1, 80, time)
            
        return mel
    
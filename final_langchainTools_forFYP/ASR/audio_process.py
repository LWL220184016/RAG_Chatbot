import queue
import numpy as np
import time
import noisereduce as nr
import torchaudio
import torch
import io
import pyaudio

from pydub import AudioSegment

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

    input_device_index: int
        For select input device, default is 1. 
        You can use pyaudio.PyAudio().get_device_info_by_index(i) to get device info.
        The index is the device index. 
        You can also use pyaudio.PyAudio().get_device_count() to get the number of devices.

    format: str
        float32 or int16
        Default is float32.
    """
    def __init__(
            self, 
            chunk: int = 512, 
            channels: int = 1, 
            rate: int = 16000, 
            audio_unchecked_queue: queue = queue.Queue(), 
            audio_checked_queue: queue = queue.Queue(), 
            format: str = "float32",
            startStream: bool = True,
            is_user_talking = None,
            stop_event = None,
            input_device_index: int = 1, 
        ):
        self.rate = rate
        self.chunk = chunk
        self.audio_unchecked_queue = audio_unchecked_queue
        self.audio_checked_queue = audio_checked_queue
        if format == "float32":
            self.paformat = pyaudio.paFloat32
            self.npformat = np.float32
            self.get_audio = self._get_audio_float32
        elif format == "int16":
            self.paformat = pyaudio.paInt16
            self.npformat = np.int16
            self.get_audio = self._get_audio_int16

        if startStream:
            self._setup_pyaudio_stream(channels, input_device_index)

        self.is_user_talking = is_user_talking
        self.stop_event = stop_event
        self.mel_spectrogram = None

    def _setup_pyaudio_stream(self, 
                              channels, 
                              input_device_index, 
                             ):

        self.stream = pyaudio.PyAudio().open(
            format=self.paformat, 
            channels=channels, 
            rate=self.rate, 
            input=True, 
            frames_per_buffer=self.chunk, 
            input_device_index=input_device_index
        )
        
    def _get_audio_int16(self, audio_data: bytes) -> np.ndarray:
        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_data
    
    def _get_audio_float32(self, audio_data: bytes) -> np.ndarray:
        audio_data = np.frombuffer(audio_data, dtype=np.float32) / 32768.0
        return audio_data
    


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
                audio_data = np.frombuffer(frame, dtype=self.npformat)

                # 計算聲音強度
                if self.npformat == np.int16:
                    # for dtype is np.int16
                    sound_volume = np.linalg.norm(audio_data) / self.chunk
                    print(f'聲音強度: {sound_volume:.2f}')
                else:
                    # for dtype is np.float32
                    sound_volume = np.sqrt(np.mean(audio_data**2))
                    sound_volume *= 40 # After calculation, the value is only a few tenths, which is too small.
                    print(f'聲音強度: {sound_volume:.2f}')
                    
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

        record_start_time = 0
        while not self.stop_event.is_set(): # 加入一個參數輸入chunk 數量或者毫秒，當聲音強度低過閾值時，等待一段時間，再檢查聲音強度
            try:
                frame = self.audio_unchecked_queue.get()
                audio_data = np.frombuffer(frame, dtype=self.npformat)
                
                # 計算聲音強度
                if self.npformat == np.int16:
                    # for dtype is np.int16
                    sound_volume = np.linalg.norm(audio_data) / self.chunk
                    print(f'聲音強度: {sound_volume:.2f}')
                else:
                    # for dtype is np.float32
                    sound_volume = np.sqrt(np.mean(audio_data**2))
                    sound_volume *= 40 # After calculation, the value is only a few tenths, which is too small.
                    print(f'聲音強度: {sound_volume:.2f}')
                
                if sound_volume > sound_level_threshold:
                    # print("Sound detected, ", f'聲音強度: {sound_volume:.2f}')
                    # print("last sound detect: ", time.time())
                    self.audio_checked_queue.put(frame)
                    record_start_time = time.time()
                    self.is_user_talking.set()

                else:
                    # if len(frames) > 0:
                        if time.time() - record_start_time < timeout_sec:
                            self.audio_checked_queue.put(frame)
                            continue
                        # frames = bytearray()
                        record_start_time = 0
                        self.is_user_talking.clear()
                        # print(f'聲音強度: {sound_volume:.2f}')
            except OSError as e:
                import traceback
                traceback.print_exc()

    def process_audio(
            self, 
            audio_data: bytes,
        ) -> np.ndarray:

        try:
            audio_data = self.get_audio(audio_data) # convert bytes to numpy array

            audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data) # normalize audio float data
            audio_data = nr.reduce_noise(y = audio_data, sr = self.rate) # reduced noise
        
            return audio_data
        except ValueError as e:
            print(f"ValueError: {e}")
            return None

    def process_audio_ws(
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
            samples = nr.reduce_noise(y = samples, sr = self.rate) # reduced noise
            # samples = np.iinfo(audio.array_type).max  # Normalized to [-1, 1]
            
            return samples
        except Exception as e:
            print(f"Processing Error: {e}")
            return None

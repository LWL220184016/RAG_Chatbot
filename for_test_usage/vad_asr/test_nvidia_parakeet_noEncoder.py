import os.path
import sys
# Add the parent folder of the current parent folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import nemo.collections.asr as nemo_asr
import pyaudio
import threading
from final.ASR.audio_process import Audio_Processer

CHUNK = 512
sampling_rate = 16000
CHANNEL = 1
FORMAT = pyaudio.paInt16
SOUND_LEVEL = 1

# asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-rnnt-1.1b")
# processer = nemo_asr.modules.AudioToMelSpectrogramPreprocessor()

if __name__ == "__main__":
    asr_model.eval()

    ap = Audio_Processer(chunk=CHUNK)

    get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
    check_audio_thread = threading.Thread(target=ap.detect_sound, args=(SOUND_LEVEL,))

    get_audio_thread.start()
    check_audio_thread.start()

    while True:
        print("Recording...")
        try:
            audio_data = ap.audio_checked_queue.get()
            print("Type of audio_data:", type(audio_data))
            print("Class of audio_data:", audio_data.__class__)

            processed_data = ap.process_audio2(audio_data=audio_data)
            print("Type of processed audio_data:", type(processed_data))
            print("Class of processed audio_data:", processed_data.__class__)
            print("processed_data: " + str(processed_data.shape))
            # processed_data = processed_data.reshape(1, 2048, 1)

            transcriptions = asr_model.transcribe(
                # encoded_features[0],
                [processed_data],
                batch_size = 4
            ) # Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` are supported as input.
            print("transcriptions: " + str(transcriptions))

        except KeyboardInterrupt:
            ap.stop_event.set()
            get_audio_thread.join()
            check_audio_thread.join()
            ap.stream.stop_stream()
            ap.stream.close()
            ap.p.terminate()
            print("User stopped the program")

        # finally:
        #     # Stop threads and clean up
        #     ap.stop_event.set()
        #     get_audio_thread.join()
        #     check_audio_thread.join()
        #     ap.stream.stop_stream()
        #     ap.stream.close()
        #     ap.p.terminate()
# import os.path
# import sys
# # Add the parent folder of the current parent folder to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

# import nemo.collections.asr as nemo_asr
# import pyaudio
# import threading
# import torch
# from nemo.collections.asr.modules import ConformerEncoder
# from final.ASR.audio_process import Audio_Processer

# CHUNK = 4096
# sampling_rate = 16000
# CHANNEL = 1
# FORMAT = pyaudio.paInt16
# SOUND_LEVEL = 1
# n_mels = 200

# # asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
# asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-rnnt-1.1b")
# processer = nemo_asr.modules.AudioToMelSpectrogramPreprocessor()
# encoder = ConformerEncoder(
#     feat_in=n_mels,  # Input feature dimension, e.g., number of Mel spectrogram features
#     n_layers=24,  # 24 Conformer blocks
#     d_model=n_mels,  # Hidden layer size
#     subsampling='striding',  # Use striding for subsampling
#     subsampling_factor=10,  # Subsampling factor of 10
#     self_attention_model='rel_pos',  # Relative positional embedding with Transformer-XL
#     n_heads=10,  # 10 attention heads
#     dropout=0.1,  # Dropout rate set to 0.1
#     dropout_att=0.1,  # Attention layer dropout rate set to 0.1
#     att_context_size=[32, 32],  # Limit context size for streaming (32 frames in past and future)
#     conv_kernel_size=129,  # Convolution kernel size
#     conv_norm_type='batch_norm',  # Use batch normalization
#     stochastic_depth_drop_prob=0.1,  # Stochastic depth drop probability
#     stochastic_depth_mode='linear',  # Stochastic depth mode is linear
# )

# encoder.setup_streaming_params(
#     chunk_size=CHUNK,
#     shift_size=CHUNK / 2,
#     left_chunks=10,
#     att_context_size=[12800, 12800],
#     max_context=10000,
# )

# if __name__ == "__main__":
#     asr_model.eval()
#     encoder.eval()

#     ap = Audio_Processer(chunk=CHUNK)
#     ap.set_mel_spectrogram(n_mels=n_mels, n_fft=1000, hop_length=800)

#     get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
#     check_audio_thread = threading.Thread(target=ap.detect_sound_not_extend, args=(SOUND_LEVEL,))

#     get_audio_thread.start()
#     check_audio_thread.start()

#     print("Recording...")
#     while True:
#         try:
#             audio_data = ap.audio_checked_queue.get()
#             # print("Type of audio_data:", type(audio_data))
#             # print("Class of audio_data:", audio_data.__class__)

#             processed_data = ap.process_audio3(audio_data=audio_data)
#             # print("Type of processed audio_data:", type(processed_data))
#             # print("Class of processed audio_data:", processed_data.__class__)
#             # print("processed_data: " + str(processed_data.shape))
#             # processed_data = processed_data.reshape(1, 2048, 1)

#             # Ensure the input tensor has more than one value per channel
            

#             encoded_output, encoded_lengths = encoder(audio_signal=processed_data, length=None) 
#             # print("Type of encoded_features:", type(encoded_output)) 
#             # print("Class of encoded_features:", encoded_lengths.__class__) 
#             # print("Type of encoded_features:", type(encoded_output)) 
#             # print("Class of encoded_features:", encoded_lengths.__class__) 
#             print("encoded_output: " + str(encoded_output.shape)) 
#             print("encoded_lengths: " + str(encoded_lengths.shape)) 
#             encoded_features = encoded_output.squeeze(0).squeeze(1)
#             print("encoded_features: " + str(encoded_features.shape)) 

#     # RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (256, 256) at dimension 2 of input [1, 1, 1]
#             transcriptions = asr_model.transcribe(
#                 encoded_features,
#                 batch_size = 1
#             ) # Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` are supported as input.
#             # transcriptions = asr_model.transcribe(["output_txt2.wav"])
#             print("transcriptions: " + str(transcriptions))

#         except KeyboardInterrupt:
#             ap.stop_event.set()
#             get_audio_thread.join()
#             check_audio_thread.join()
#             ap.stream.stop_stream()
#             ap.stream.close()
#             ap.p.terminate()
#             print("User stopped the program")

#         finally:
#             # Stop threads and clean up
#             ap.stop_event.set()
#             get_audio_thread.join()
#             check_audio_thread.join()
#             ap.stream.stop_stream()
#             ap.stream.close()
#             ap.p.terminate()


import os.path
import sys
# Add the parent folder of the current parent folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import nemo.collections.asr as nemo_asr
import pyaudio
import threading
import torch
from nemo.collections.asr.modules import ConformerEncoder
from final.ASR.audio_process import Audio_Processer

CHUNK = 1024
sampling_rate = 16000
CHANNEL = 1
FORMAT = pyaudio.paInt16
SOUND_LEVEL = 1
n_mels = 400

asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-rnnt-1.1b")
processer = nemo_asr.modules.AudioToMelSpectrogramPreprocessor()
encoder = ConformerEncoder(
    feat_in=n_mels,  # Input feature dimension, e.g., number of Mel spectrogram features
    n_layers=24,  # 24 Conformer blocks
    d_model=n_mels,  # Hidden layer size
    subsampling='striding',  # Use striding for subsampling
    subsampling_factor=10,  # Subsampling factor of 10
    self_attention_model='rel_pos',  # Relative positional embedding with Transformer-XL
    n_heads=8,  # 10 attention heads
    dropout=0.1,  # Dropout rate set to 0.1
    dropout_att=0.1,  # Attention layer dropout rate set to 0.1
    att_context_size=[32, 32],  # Limit context size for streaming (32 frames in past and future)
    conv_kernel_size=65,  # Convolution kernel size
    conv_norm_type='batch_norm',  # Use batch normalization
    stochastic_depth_drop_prob=0.1,  # Stochastic depth drop probability
    stochastic_depth_mode='linear',  # Stochastic depth mode is linear
)

encoder.setup_streaming_params(
    chunk_size=CHUNK,
    shift_size=CHUNK / 2,
    left_chunks=10,
    att_context_size=[12800, 12800],
    max_context=10000,
)

if __name__ == "__main__":
    asr_model.eval()
    encoder.eval()

ap = Audio_Processer(chunk=CHUNK)
ap.set_mel_spectrogram(n_mels=n_mels)

get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
check_audio_thread = threading.Thread(target=ap.detect_sound_not_extend, args=(SOUND_LEVEL,))

get_audio_thread.start()
check_audio_thread.start()

print("Recording...")
while True:
    try:
        audio_data = ap.audio_checked_queue.get()
        print("Type of audio_data:", type(audio_data))
        print("Class of audio_data:", audio_data.__class__)

        processed_data = ap.process_audio3(audio_data=audio_data)
        print("Type of processed audio_data:", type(processed_data))
        print("Class of processed audio_data:", processed_data.__class__)
        print("processed_data: " + str(processed_data.shape))
        # processed_data = processed_data.reshape(1, 2048, 1)

        # Ensure the input tensor has more than one value per channel
        

        encoded_output, encoded_lengths = encoder(audio_signal=processed_data, length=None) 
        print("Type of encoded_features:", type(encoded_output)) 
        print("Class of encoded_features:", encoded_lengths.__class__) 
        print("Type of encoded_features:", type(encoded_output)) 
        print("Class of encoded_features:", encoded_lengths.__class__) 
        print("encoded_output: " + str(encoded_output.shape)) 
        print("encoded_lengths: " + str(encoded_lengths.shape)) 
        # print("processed_data: " + str(encoded_features)) 
        encoded_features = encoded_output.squeeze(0).squeeze(1)
        print("encoded_features: " + str(encoded_features.shape)) 

# RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (256, 256) at dimension 2 of input [1, 1, 1]
        transcriptions = asr_model.transcribe(
            # encoded_features[0],
            [encoded_features],
            batch_size = 4
        ) # Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` are supported as input.
        # transcriptions = asr_model.transcribe(["output_txt2.wav"])
        print("transcriptions: " + str(transcriptions))

    except KeyboardInterrupt:
        ap.stop_event.set()
        get_audio_thread.join()
        check_audio_thread.join()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
        print("User stopped the program")

    finally:
        # Stop threads and clean up
        ap.stop_event.set()
        get_audio_thread.join()
        check_audio_thread.join()
        ap.stream.stop_stream()
        ap.stream.close()
        ap.p.terminate()
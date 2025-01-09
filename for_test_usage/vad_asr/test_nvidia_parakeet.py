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

CHUNK = 2048
sampling_rate = 16000
CHANNEL = 1
FORMAT = pyaudio.paInt16
SOUND_LEVEL = 20
n_mels = 80

# asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-rnnt-1.1b")
# processer = nemo_asr.modules.AudioToMelSpectrogramPreprocessor()
encoder = ConformerEncoder(
    feat_in=n_mels,  # 输入特征维度，例如 Mel 频谱图的特征数
    n_layers=12,  # 12 个 Conformer 块
    d_model=n_mels,  # 隐藏层大小
    subsampling='striding',  # 使用 striding 作为子采样方法
    subsampling_factor=8,  # 子采样因子为4
    self_attention_model='rel_pos',  # 相对位置嵌入和 Transformer-XL
    n_heads=8,  # 多头注意力使用8个头
    dropout=0.1,  # dropout 率设置为0.1
    dropout_att=0.1,  # 注意力层的 dropout 率设置为0.1
    att_context_size=[-1, -1],  # 无限上下文（用于非流式情况）
    conv_kernel_size=129,  # 卷积核大小
    conv_norm_type='batch_norm',  # 使用批归一化
    stochastic_depth_drop_prob=0.1,  # 随机深度丢弃概率
    stochastic_depth_mode='linear'  # 随机深度模式为线性
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
    ap.set_mel_spectrogram()

    get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
    check_audio_thread = threading.Thread(target=ap.detect_sound_not_extend, args=(SOUND_LEVEL,))

    get_audio_thread.start()
    check_audio_thread.start()

    print("Recording...")
    # while True:
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
        

        encoded_features = encoder(audio_signal=processed_data, length=None) 
        print("Type of encoded_features:", type(encoded_features))
        print("Class of encoded_features:", encoded_features.__class__)
        # print("processed_data: " + str(encoded_features))


RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (256, 256) at dimension 2 of input [1, 1, 1]
        transcriptions = asr_model.transcribe([encoded_features[1]]) # Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` are supported as input.
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
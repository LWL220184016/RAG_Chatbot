import pyaudio
import numpy as np

# 設定參數
FORMAT = pyaudio.paInt16  # 音頻格式
CHANNELS = 1              # 單聲道
RATE = 16000              # 取樣率
CHUNK = 1024              # 每次讀取的樣本數

# 初始化 PyAudio
p = pyaudio.PyAudio()

# 開啟音訊流
stream = p.open(format=FORMAT, channels=CHANNELS,
                 rate=RATE, input=True,
                 frames_per_buffer=CHUNK)

print("開始錄音，請說話...")

try:
    while True:
        # 讀取音訊數據
        data = stream.read(CHUNK)
        # 將數據轉換為 numpy 陣列
        audio_data = np.frombuffer(data, dtype=np.int16)
        # 計算聲音強度
        volume_norm = np.linalg.norm(audio_data) / CHUNK
        print(f'聲音強度: {volume_norm:.2f}')
except KeyboardInterrupt:
    print("錄音結束。")
finally:
    # 停止和關閉流
    stream.stop_stream()
    stream.close()
    p.terminate()
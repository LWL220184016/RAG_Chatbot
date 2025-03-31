import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

# --- 設定 ---
TARGET_DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)" # <--- 請根據上一步驟的結果修改此名稱！
# 或者，如果你知道索引，可以使用索引：
# TARGET_DEVICE_ID = 5 # <--- 範例索引，請修改！

SAMPLE_RATE = 44100  # 取樣率 (Hz)，常見值 44100 或 48000
DURATION = 10      # 錄音時長 (秒)
FILENAME = "system_audio_output.wav" # 儲存的檔案名

# --- 尋找裝置 ---
device_id = None
devices = sd.query_devices()
found = False
for i, device in enumerate(devices):
    # 我們要找的是 *輸入* (input) 裝置，因為我們要從它錄音
    if TARGET_DEVICE_NAME in device['name'] and device['max_input_channels'] > 0:
         device_id = i
         print(f"找到裝置: '{device['name']}' (索引: {device_id})")
         found = True
         break

# 如果使用裝置 ID，取消下面這行的註解並註解掉上面的尋找區塊
# device_id = TARGET_DEVICE_ID
# found = True # 假設 ID 是正確的

if not found:
    print(f"錯誤：找不到名為 '{TARGET_DEVICE_NAME}' 的輸入裝置。")
    print("可用的裝置:")
    print(sd.query_devices())
    exit()

# --- 錄音 ---
print(f"準備從 '{TARGET_DEVICE_NAME}' 錄音 {DURATION} 秒...")
try:
    # 查詢裝置支援的聲道數
    device_info = sd.query_devices(device_id, 'input')
    channels = 2
    # 通常是 2 (立體聲)，如果查詢失敗或需要指定，可以設 channels=2
    print(f"使用 {channels} 聲道，取樣率 {SAMPLE_RATE} Hz")

    # 開始錄音
    # dtype='float32' 通常效果較好，也可以用 'int16'
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=channels, device=device_id, dtype='float32')

    # 等待錄音完成 (可以加入進度提示)
    for i in range(DURATION):
        print(f"錄音中... {DURATION - i} 秒剩餘")
        time.sleep(1)
    sd.wait() # 確保錄音完全結束

    print("錄音完成。")

    # --- 儲存檔案 ---
    print(f"正在儲存為 {FILENAME}...")

    # 將 float32 數據轉換為 int16 以便儲存為標準 WAV 格式
    # 乘以 32767 並轉換類型
    wav_data = np.int16(recording * 32767)
    write(FILENAME, SAMPLE_RATE, wav_data)

    # 如果想直接儲存 float32 的 WAV (不是所有播放器都支援):
    # write(FILENAME, SAMPLE_RATE, recording)

    print(f"錄音已成功儲存至 {FILENAME}")

except sd.PortAudioError as e:
    print(f"錄音時發生 PortAudio 錯誤: {e}")
    print("請檢查：")
    print("1. VB-CABLE 是否正確安裝並運作？")
    print("2. '聆聽此裝置' 是否已按上述步驟設定？")
    print("3. 裝置名稱/索引是否正確？")
    print(f"4. 裝置 '{TARGET_DEVICE_NAME}' 是否支援 {SAMPLE_RATE} Hz 取樣率和 {channels} 聲道？")
except Exception as e:
    print(f"發生未預期的錯誤: {e}")
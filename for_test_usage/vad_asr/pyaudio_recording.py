import wave
import sys
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 20

# Find the index of the "Stereo Mix" device
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

with wave.open('output.wav', 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=stereo_mix_index)

    print('Recording...')
    for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
        wf.writeframes(stream.read(CHUNK))
    print('Done')

    stream.close()
    p.terminate()


# # Using microphone to record audio and save it to a file
# import wave
# import sys

# import pyaudio

# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1 if sys.platform == 'darwin' else 2
# RATE = 44100
# RECORD_SECONDS = 5

# with wave.open('output.wav', 'wb') as wf:
#     p = pyaudio.PyAudio()
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)

#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

#     print('Recording...')
#     for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
#         wf.writeframes(stream.read(CHUNK))
#     print('Done')

#     stream.close()
#     p.terminate()



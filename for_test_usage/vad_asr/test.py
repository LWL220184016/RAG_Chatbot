from for_test_usage.vad_asr.asr_file import get_asr_model
# from vad_file import record_and_detect_vad
from vad_system import record_and_detect_vad, get_audio_chunk
import queue
import threading
import pyaudio
import sys

CHUNK = 512  # Reduce chunk size to avoid overflow
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

if __name__ == '__main__':
    stereo_mix_index = None
    print(p.get_device_count())
    asr_model, asr_processor, device, torch_dtype = get_asr_model()

    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if "Stereo Mix" in dev_info.get('name', ''):
            stereo_mix_index = i
            break

    if stereo_mix_index is None:
        print("Stereo Mix device not found")
        sys.exit(1)

    start = False
    start_time = None
    audio_chunk_queue = queue.Queue(50)
    sample_rate = 16000
    stop_event = threading.Event()

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=stereo_mix_index, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk, args=(stream, audio_chunk_queue, stop_event))
        get_audio_threading.start()
        for frame in record_and_detect_vad(audio_chunk_queue, stream):
            print("Recording...")
            input_features = asr_processor(
                frame, sampling_rate=16000, return_tensors="pt"
            )
            input_features = input_features.to(device, dtype=torch_dtype)
            # print(len(frame))

            pred_ids = asr_model.generate(input_features.input_features)
            pred_text = asr_processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

            print(pred_text)
    except KeyboardInterrupt:
        print("Recording stopped by user")
    finally:
        stop_event.set()
        get_audio_threading.join()
        if stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

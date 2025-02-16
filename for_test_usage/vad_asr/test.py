from asr_file import get_asr_model
# from vad_file import record_and_detect_vad
from vad_system import record_and_detect_vad
import queue
import threading
import pyaudio
import sys
from check_input_device import get_input_device, get_audio_chunk

CHUNK = 512  # Reduce chunk size to avoid overflow
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


if __name__ == '__main__':
    p = pyaudio.PyAudio()
    sound_device_index = get_input_device(p, "CABLE Output")
    asr_model, asr_processor, device, torch_dtype = get_asr_model()

    start = False
    start_time = None
    audio_chunk_queue = queue.Queue(50)
    sample_rate = 16000
    stop_event = threading.Event()

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=sound_device_index, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk, args=(stream, audio_chunk_queue, stop_event, CHUNK))
        get_audio_threading.start()

        for frame in record_and_detect_vad(audio_chunk_queue, stream):
            if not stream.is_active():
                break
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

import pyaudio
import threading
import queue
import sounddevice as sd
from for_test_usage.vad_asr.check_input_device import get_audio_chunk_sound_level
from for_test_usage.vad_asr.asr_system import convert_audio_bytes_to_float, normalize_audio, get_asr_model
from for_test_usage.llm_tts.llm_tts import get_LLM, get_TTS, llm_output, tts_output

SOUND_LEVEL = 100
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SEC = 1

if __name__ == "__main__":
    llm = get_LLM()
    processor, model, vocoder, speaker_embeddings = get_TTS()
    p = pyaudio.PyAudio()
    # sound_device_index = get_input_device(p, "Microphone (MONSTER AIRMARS N3)")

    fileNum = 0
    audio_chunk_queue = queue.Queue(50)
    prompt_queue = queue.Queue()
    output_queue = queue.Queue()
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    speaking_event = threading.Event()

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        get_audio_threading = threading.Thread(target=get_audio_chunk_sound_level, args=(stream, audio_chunk_queue, stop_event, CHUNK, SOUND_LEVEL))
        llm_thread = threading.Thread(target=llm_output, args=(llm, prompt_queue, output_queue))
        tts_thread = threading.Thread(target=tts_output, args=(processor, model, vocoder, speaker_embeddings, output_queue, audio_queue, speaking_event))
        get_audio_threading.start()
        llm_thread.start()
        tts_thread.start()
        asr_model, asr_processor, device, torch_dtype = get_asr_model()

        while True:
            print("Recording...")
            frame = audio_chunk_queue.get()
            audio_data = convert_audio_bytes_to_float(frame)

            # audio_data = normalize_audio(audio_data)
            input_features = asr_processor(
                audio_data, sampling_rate=RATE, return_tensors="pt"
            )
            input_features = input_features.to(device, dtype=torch_dtype)
            # print(len(frame))

            pred_ids = asr_model.generate(input_features.input_features)
            prompt = asr_processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

            print("\nYou: " + prompt)

            # prompt = input("You: ")
            prompt_queue.put(prompt)
            speaking_event.set()  # Signal that LLM is speaking
            
            while speaking_event.is_set() or not audio_queue.empty():
                audio_chunk = audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=True)
                sd.wait()
                audio_queue.task_done()
    except KeyboardInterrupt:
        stop_event.set()
        get_audio_threading.join()
        stream.stop_stream()
        stream.close()
        p.terminate()
import collections
import torch
import numpy as np
from silero_vad import get_speech_timestamps, read_audio, VADIterator
# from vad.vad import VADIterator

CHUNK = 512
CHANNELS = 2
RATE = 44100

# copy from https://github.com/huggingface/speech-to-speech
def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f for f in voiced_frames])

def record_and_detect_vad(file_path):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    vad = VADIterator(model)
    sample_rate = 16000
    frame_duration = 30  # ms
    padding_duration = 300  # ms

    frames = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        frames.append(indata.copy())

    # try:
    wav = read_audio(file_path, sampling_rate=sample_rate)
    return_chunk = False
    window_size_samples = 512 if sample_rate == 16000 else 256
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i+ window_size_samples]
        if len(chunk) < window_size_samples:
            break

        speech_dict = vad(chunk, return_seconds=True)
        if speech_dict and return_chunk == True:
            return_chunk = False
        if speech_dict or return_chunk == True:
            yield speech_dict, chunk
            return_chunk = True
        
    vad.reset_states() # reset model states after each audio

    # except Exception as e:
    #     print(f"An error occurred: {e}")

if __name__ == '__main__':
    for speech_dict, chunk in record_and_detect_vad("for_test_usage/vad_asr/output.wav"):
        print(speech_dict)
        
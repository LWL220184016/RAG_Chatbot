import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import asyncio
import websockets
import multiprocessing
import sounddevice as sd
from final_test.TTS.tts import TTS


async def echo(websocket, llm_output_queue):
    async for message in websocket:
        print(f"Received message: {message}")
        llm_output_queue.put(message)
        await websocket.send(f"{message}")

def tts_process_func(stop_event, llm_output_queue, speaking_event, audio_queue):
    try:
        tts = TTS(stop_event=stop_event, audio_queue=audio_queue)
        print("waiting llm output")
        tts.tts_output(llm_output_queue, speaking_event)
    finally:
        stop_event.set()

def play_audio(stop_event, audio_queue):
    print("waiting audio")
    while not stop_event.is_set():
        audio_chunk = audio_queue.get()
        sd.play(audio_chunk, samplerate=16000, blocking=False)
        sd.wait()

async def main(llm_output_queue):
    async with websockets.serve(lambda ws: echo(ws, llm_output_queue), "localhost", 6789):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    speaking_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()
    llm_output_queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()
    tts_process = multiprocessing.Process(target=tts_process_func, args=(stop_event, llm_output_queue, speaking_event, audio_queue))
    play_audio_process = multiprocessing.Process(target=play_audio, args=(stop_event, audio_queue))
    tts_process.start()
    play_audio_process.start()

    # Create and run a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main(llm_output_queue))

    
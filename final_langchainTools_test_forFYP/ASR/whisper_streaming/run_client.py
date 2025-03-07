#!/usr/bin/env python3
import asyncio
import websockets
import json
import argparse
import sounddevice as sd
import numpy as np

async def send_audio_stream(uri, input_device=None, sample_rate=16000):
    """Stream audio from microphone to ASR server via WebSocket"""
    print(f"Connecting to ASR server at {uri}")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Start speaking...")
        
        # Audio queue for handling microphone data
        audio_queue = asyncio.Queue()
        
        # Callback function for the microphone stream
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            # Convert audio to mono and int16 format
            audio_mono = indata[:, 0] if indata.shape[1] > 1 else indata[:, 0]
            # Scale to int16 range and convert directly to bytes
            audio_int16 = (audio_mono * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Add to queue for processing
            asyncio.run_coroutine_threadsafe(audio_queue.put(audio_bytes), loop)
        
        # Get the event loop
        loop = asyncio.get_event_loop()
        
        # Start the stream task
        stream_task = asyncio.create_task(process_audio_queue(websocket, audio_queue))
        
        # Start receiving results task
        receive_task = asyncio.create_task(receive_results(websocket))
        
        # Start audio capture with sounddevice
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(sample_rate * 0.1),  # 100ms chunks
            device=input_device
        ):
            print("Recording... Press Ctrl+C to stop.")
            try:
                await asyncio.gather(stream_task, receive_task)
            except KeyboardInterrupt:
                print("Stopping...")
                # Send finish command to get final transcription
                await websocket.send(json.dumps({"command": "finish"}))
                # Wait for final results
                await asyncio.sleep(1)

async def process_audio_queue(websocket, queue):
    """Process and send audio chunks from queue"""
    try:
        while True:
            # Get audio data from queue
            audio_data = await queue.get()
            
            # Send binary audio data to server
            await websocket.send(audio_data)
            
            # Small delay to prevent overwhelming server
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        print("Audio streaming stopped")

async def receive_results(websocket):
    """Receive and print transcription results"""
    try:
        async for message in websocket:
            try:
                result = json.loads(message)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    text = result.get("text", "")
                    start = result.get("start", 0) / 1000
                    end = result.get("end", 0) / 1000
                    final = result.get("final", False)
                    
                    status = "[FINAL]" if final else "[partial]"
                    print(f"{status} [{start:.2f}s - {end:.2f}s]: {text}")
            except json.JSONDecodeError:
                print(f"Received invalid JSON: {message}")
    except asyncio.CancelledError:
        print("Result receiving stopped")

def main():
    parser = argparse.ArgumentParser(description="WebSocket ASR Client Example")
    parser.add_argument("--server", type=str, default="ws://localhost:8765",
                        help="WebSocket server URI")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device ID (see --list-devices)")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("Available audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{i}] {device['name']}")
        return
    
    try:
        asyncio.run(send_audio_stream(args.server, args.device))
    except KeyboardInterrupt:
        print("Program terminated by user")

if __name__ == "__main__":
    main()

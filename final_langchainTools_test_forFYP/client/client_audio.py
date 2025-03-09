import asyncio
import threading
import websockets
import json
import argparse
import sounddevice as sd
import numpy as np
import queue
import time
import base64

from audio_process import Audio_Processer

user_color_code = 155 # 顔色
llm_color_code = 202  # 顔色

async def send_audio_stream(uri, input_device=None, sample_rate=16000):
    """Stream audio from microphone to ASR server via WebSocket"""
    print(f"Connecting to ASR server at {uri}")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Start speaking...")

        audio_checked_queue = queue.Queue()
        tts_audio_queue = queue.Queue()

        is_user_talking = threading.Event()
        stop_event = threading.Event()
        
        ap = Audio_Processer( 
            chunk = 512, 
            audio_checked_queue = audio_checked_queue, 
            is_user_talking = is_user_talking, 
            stop_event = stop_event, 
            input_device_index = input_device,
        ) 
        
        # Create a shared event to signal when we need to stop threads
        thread_stop_event = threading.Event()
        
        try:
            get_audio_thread = threading.Thread(target=ap.get_chunk, args=(True,))
            check_audio_thread = threading.Thread(target=ap.detect_sound, args=(10, 0.3))
            
            # Use the non-async version of send_audio for threading
            send_audio_thread = threading.Thread(
                target=send_audio, 
                args=(websocket, audio_checked_queue, thread_stop_event, )
            )
            play_tts_audio_thread = threading.Thread(
                target=play_tts_audio, 
                args=(tts_audio_queue, )
            )
            
            get_audio_thread.daemon = True
            check_audio_thread.daemon = True
            send_audio_thread.daemon = True
            play_tts_audio_thread.daemon = True

            get_audio_thread.start()
            check_audio_thread.start()
            send_audio_thread.start()
            play_tts_audio_thread.start()

            # Add receive_messages task to handle messages from the server
            receive_messages_task = asyncio.create_task(receive_messages(websocket, tts_audio_queue))
            
            # Run all tasks concurrently
            await asyncio.gather(
                receive_messages_task, 
            )

        except json.JSONDecodeError:
            print(f"Received invalid JSON message")

        except Exception as e:
            import traceback
            traceback.print_exc()

        except KeyboardInterrupt:
            print("Stopping...")
            # Send finish command to get final transcription
            await websocket.send(json.dumps({"command": "finish"}))
            # Wait for final results
            await asyncio.sleep(1)
        finally:
            # Signal threads to stop
            thread_stop_event.set()
            # Ensure to stop the event so tasks can clean up properly
            stop_event.set()
            get_audio_thread.join()
            check_audio_thread.join()
            send_audio_thread.join()
            play_tts_audio_thread.join()
            
            # Give threads a moment to clean up
            time.sleep(0.5)

async def receive_messages(websocket, tts_audio_queue):
    """Receive and process messages from the server"""
    try:
        async for message in websocket:
            if message.startswith("AUDIO: "):
                audio_data = message[7:]
                tts_audio_queue.put(audio_data)

            elif message.startswith("You: "):
                print(f"\n\033[38;5;{user_color_code}m{message}\033[0m")

            elif message.startswith("LLM: "):
                print(f"\n\033[38;5;{llm_color_code}m{message}\033[0m")

    except Exception as e:
        print(f"Error receiving messages: {e}")

# This version of send_audio is for async contexts
async def send_audio(websocket, audio_queue):
    """Process and send audio chunks from queue - async version"""
    try:
        while True:
            # Get audio data from queue
            try:
                audio_data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Send binary audio data to server
            await websocket.send(audio_data)  # Use await here
            print("Send audio data")
            # Small delay to prevent overwhelming server
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        print("Audio streaming stopped")

def play_tts_audio(tts_audio_queue):
    """Play audio chunks from queue"""
    print("waiting for audio data...")
    try:
        while True:
            # Get audio data from queue
            try:
                audio_data = tts_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            audio_data = base64.b64decode(audio_data)
            # Play audio data
            print("Playing audio...")
            sd.play(np.frombuffer(audio_data, dtype=np.int16), samplerate=16000)
            sd.wait()
    except asyncio.CancelledError:
        print("Audio playback stopped")

# Non-async version for threads
def send_audio(websocket, audio_queue, stop_event):
    """Process and send audio chunks from queue - thread version"""
    try:
        while not stop_event.is_set():
            # Get audio data from queue
            try:
                audio_data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Use the synchronous API via run_until_complete
            asyncio.run(websocket.send(audio_data))
            print("Send audio data")
            # Small delay to prevent overwhelming server
            time.sleep(0.05)
    except Exception as e:
        print(f"Error sending audio: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="WebSocket ASR Client Example")
    parser.add_argument("--server", type=str, default="ws://localhost:6789",
                        help="WebSocket server URI")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--device", type=int, default=1,
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
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
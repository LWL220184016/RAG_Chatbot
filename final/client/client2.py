import asyncio
import threading
import websockets
import json
import argparse
import sounddevice as sd
import numpy as np
import queue  # Standard queue for thread-to-thread
import asyncio # Import asyncio queue for async/thread bridge
import time
import base64
import logging
from typing import Optional, Any # For type hints

# Assuming Audio_Processer is in a local file named audio_process.py
# If not, you MUST place the Audio_Processer class definition here or import it
try:
    from audio_process import Audio_Processer
except ImportError:
    print("Error: Could not import Audio_Processer from audio_process.py")
    print("Please ensure the file exists and the class is defined correctly.")
    # exit(1) # Optional: force exit if class is missing

# --- Configuration ---
USER_COLOR_CODE = 155
LLM_COLOR_CODE = 202
DEFAULT_SAMPLE_RATE = 16000
WEBSOCKET_TIMEOUT = 10 # Timeout for websocket connection/messages
THREAD_JOIN_TIMEOUT = 2 # Timeout for waiting for threads to join

# --- Logging ---
# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress overly verbose logs from libraries if needed
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("sounddevice").setLevel(logging.WARNING)


# --- Asynchronous Tasks ---

async def receive_messages_task(
    websocket: websockets.WebSocketClientProtocol,
    tts_audio_queue: queue.Queue, # TTS uses standard queue for thread
    shutdown_event: asyncio.Event
):
    """Receive and process messages from the server."""
    logging.info("Receiver task started.")
    try:
        while not shutdown_event.is_set():
            try:
                # Add a timeout to websocket.recv to periodically check shutdown_event
                message_str = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue # No message received, check shutdown_event and loop again
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed by server.")
                break # Exit loop if connection closed

            try:
                # Attempt to parse as JSON first (optional, depends on server protocol)
                # data = json.loads(message_str)
                # logging.info(f"Received JSON: {data}")
                # Process JSON data...

                # Simple string prefix checking based on original code
                if isinstance(message_str, str):
                    if message_str.startswith("AUDIO: "):
                        audio_data_b64 = message_str[7:]
                        if audio_data_b64:
                            tts_audio_queue.put(audio_data_b64)
                        else:
                            logging.warning("Received empty audio data.")
                    elif message_str.startswith("You: "):
                        print(f"\n\033[38;5;{USER_COLOR_CODE}m{message_str}\033[0m", flush=True)
                    elif message_str.startswith("LLM: "):
                        print(f"\n\033[38;5;{LLM_COLOR_CODE}m{message_str}\033[0m", flush=True)
                    else:
                         # Handle other text messages if needed
                         logging.info(f"Received Text: {message_str}")
                elif isinstance(message_str, bytes):
                     # Handle binary messages if needed
                     logging.info(f"Received Binary Data: {len(message_str)} bytes")
                else:
                    logging.warning(f"Received unexpected message type: {type(message_str)}")

            except json.JSONDecodeError:
                logging.warning(f"Received non-JSON message: {message_str}")
            except Exception as e:
                logging.error(f"Error processing received message: {e}", exc_info=True)

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("WebSocket connection closed gracefully.")
    except asyncio.CancelledError:
        logging.info("Receiver task cancelled.")
    except Exception as e:
        logging.error(f"Error in receiver task: {e}", exc_info=True)
    finally:
        logging.info("Receiver task finished.")
        # Signal shutdown if not already set (e.g., if connection closed unexpectedly)
        if not shutdown_event.is_set():
            logging.warning("Receiver task ended unexpectedly, initiating shutdown.")
            shutdown_event.set()

async def send_audio_task(
    websocket: websockets.WebSocketClientProtocol,
    audio_to_send_queue: asyncio.Queue, # Use asyncio.Queue here
    shutdown_event: asyncio.Event
):
    """Get audio chunks from the asyncio queue and send them via WebSocket."""
    logging.info("Sender task started.")
    try:
        while not shutdown_event.is_set():
            try:
                # Wait for audio data from the queue with a timeout
                audio_data = await asyncio.wait_for(audio_to_send_queue.get(), timeout=0.5)
                if audio_data is None: # Use None as a signal to stop gracefully if needed
                    logging.info("Received None sentinel in send queue, stopping sender.")
                    break

                await websocket.send(audio_data)
                # logging.debug(f"Sent audio data chunk: {len(audio_data)} bytes") # Debug level
                audio_to_send_queue.task_done()

                # Small sleep to yield control, prevent tight loop if queue fills fast
                await asyncio.sleep(0.01)

            except asyncio.TimeoutError:
                continue # No data in queue, check shutdown_event and loop
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed while sending.")
                break
            except asyncio.CancelledError:
                logging.info("Sender task cancelled.")
                break # Exit loop on cancellation
            except Exception as e:
                logging.error(f"Error sending audio: {e}", exc_info=True)
                # Decide whether to break or continue based on the error
                if isinstance(e, (ConnectionError, BrokenPipeError)):
                     break
                await asyncio.sleep(0.1) # Avoid spamming logs on persistent errors

    except Exception as e:
         # Catch potential errors during initial setup or final cleanup
         logging.error(f"Unhandled error in sender task: {e}", exc_info=True)
    finally:
        logging.info("Sender task finished.")
        # Signal shutdown if not already set
        if not shutdown_event.is_set():
            logging.warning("Sender task ended unexpectedly, initiating shutdown.")
            shutdown_event.set()

async def bridge_queue_task(
    thread_queue: queue.Queue, # Source: standard queue filled by threads
    async_queue: asyncio.Queue, # Destination: asyncio queue consumed by sender task
    shutdown_event: asyncio.Event
):
    """Moves items from a thread-safe queue to an asyncio queue."""
    logging.info("Queue bridge task started.")
    loop = asyncio.get_running_loop()
    try:
        while not shutdown_event.is_set():
            try:
                # Use run_in_executor to safely get from the blocking queue
                audio_data = await loop.run_in_executor(
                    None, # Use default executor (ThreadPoolExecutor)
                    thread_queue.get, # Blocking function to run
                    True, # block=True
                    0.5 # timeout
                )
                if audio_data:
                    await async_queue.put(audio_data)
                    # logging.debug("Bridged audio chunk to async queue.") # Debug level
                thread_queue.task_done() # Signal completion to the thread queue
            except queue.Empty:
                continue # Timeout occurred, check shutdown_event and loop
            except asyncio.CancelledError:
                logging.info("Queue bridge task cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in queue bridge: {e}", exc_info=True)
                await asyncio.sleep(0.1) # Avoid tight loop on error

    except Exception as e:
         logging.error(f"Unhandled error in queue bridge task: {e}", exc_info=True)
    finally:
        logging.info("Queue bridge task finished.")
        # Optionally put a sentinel value to signal the sender task if needed
        # await async_queue.put(None)
        # Signal shutdown if not already set
        if not shutdown_event.is_set():
            logging.warning("Queue bridge task ended unexpectedly, initiating shutdown.")
            shutdown_event.set()

# --- Thread Functions ---

def play_tts_audio_thread(
    tts_audio_queue: queue.Queue,
    stop_event: threading.Event # Use threading.Event for threads
):
    """Play audio chunks from queue in a separate thread."""
    logging.info("TTS playback thread started.")
    while not stop_event.is_set():
        try:
            # Get audio data from queue with timeout
            audio_data_b64 = tts_audio_queue.get(timeout=0.5)
            if not audio_data_b64: # Skip if empty for some reason
                tts_audio_queue.task_done()
                continue

            try:
                audio_data = base64.b64decode(audio_data_b64)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                if audio_array.size > 0:
                    logging.info(f"Playing {len(audio_data)} bytes of audio...")
                    sd.play(audio_array, samplerate=DEFAULT_SAMPLE_RATE)
                    sd.wait() # Wait for playback to finish for this chunk
                    logging.info("Playback finished.")
                else:
                    logging.warning("Decoded audio data is empty, skipping playback.")

            except base64.binascii.Error:
                logging.error("Invalid Base64 data received for TTS.")
            except Exception as e:
                logging.error(f"Error during TTS playback: {e}", exc_info=True)
            finally:
                 tts_audio_queue.task_done() # Ensure task_done is called

        except queue.Empty:
            continue # No audio in queue, check stop_event and loop
        except Exception as e:
            # Catch broader exceptions in the loop
            logging.error(f"Error in TTS playback loop: {e}", exc_info=True)
            time.sleep(0.1) # Avoid tight loop on error

    logging.info("TTS playback thread finished.")

def start_audio_processing_threads(
    ap: Audio_Processer,
    stop_event: threading.Event,
    audio_chunk_queue: queue.Queue, # The queue AP writes to
    sound_level: int = 100 # Sound level threshold for detection
) -> list[threading.Thread]:
    """Starts the threads managed by the Audio_Processer."""
    threads = []
    try:
        # Ensure the queue used by AP is the one we read from
        ap.audio_checked_queue = audio_chunk_queue
        ap.stop_event = stop_event # Make sure AP uses the main stop event

        # Adapt args based on your Audio_Processer's method signatures
        # Original code had ap.get_chunk(True,) - assuming True means start immediately
        get_audio_thread = threading.Thread(
            target=ap.get_chunk,
            args=(True,), # Pass necessary args
            name="AudioChunkGetter",
            daemon=True
        )
        # Original: ap.detect_sound_not_extend(5, -1)
        check_audio_thread = threading.Thread(
            target=ap.detect_sound_not_extend,
            args=(sound_level, -1), # Pass necessary args
            name="AudioSoundDetector",
            daemon=True
        )

        threads.extend([get_audio_thread, check_audio_thread])

        for t in threads:
            t.start()
        logging.info("Audio processing threads started.")

    except AttributeError as e:
         logging.error(f"Audio_Processer is missing expected attribute/method: {e}. Cannot start threads.", exc_info=True)
         # Stop any threads that might have started if error occurs mid-way (unlikely here)
         stop_event.set()
         threads = [] # Clear threads list as they failed to start properly
    except Exception as e:
        logging.error(f"Failed to start audio processing threads: {e}", exc_info=True)
        stop_event.set()
        threads = []

    return threads

# --- Main Client Logic ---

async def run_client(
    uri: str,
    input_device: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE
):
    """Connects to the server, sets up queues, tasks, and threads."""
    logging.info(f"Attempting to connect to WebSocket server at {uri}")
    logging.info(f"Using input device ID: {input_device}, Sample Rate: {sample_rate}")

    # Queues
    # 1. Queue for Audio_Processer threads to put processed audio chunks into
    audio_from_processor_queue = queue.Queue(maxsize=50) # Thread-safe queue
    # 2. Queue for the async sender task to read from (bridged from #1)
    audio_to_send_queue = asyncio.Queue(maxsize=50) # Async queue
    # 3. Queue for received TTS audio data (read by playback thread)
    tts_audio_queue = queue.Queue(maxsize=20) # Thread-safe queue

    # Events for coordination
    # Use asyncio.Event for coordinating async tasks and signaling threads via checks
    master_shutdown_event = asyncio.Event()
    # Use threading.Event specifically for threads that might need blocking waits
    thread_stop_event = threading.Event() # For Audio_Processer and TTS thread

    # Create Audio Processor instance
    ap: Optional[Audio_Processer] = None
    audio_processor_threads: list[threading.Thread] = []
    tts_playback_thread: Optional[threading.Thread] = None
    tasks: list[asyncio.Task] = []

    try:
        ap = Audio_Processer(
            # chunk=512, # Example chunk size
            chunk = 4096,
            audio_checked_queue=audio_from_processor_queue, # Pass the correct queue
            is_user_talking=threading.Event(), # Let AP manage its own internal talking event if needed
            stop_event=thread_stop_event, # Pass the thread stop event
            input_device_index=input_device,
        )
        logging.info("Audio_Processer initialized.")
    except NameError:
         logging.error("Audio_Processer class is not defined. Please fix the import or definition.")
         return # Cannot continue without the processor class
    except Exception as e:
        logging.error(f"Failed to initialize Audio_Processer: {e}", exc_info=True)
        return # Cannot continue

    websocket_connection: Optional[websockets.WebSocketClientProtocol] = None
    try:
        # Establish WebSocket connection with timeout and ping interval
        websocket_connection = await websockets.connect(
            uri,
            ping_interval=20, # Send pings every 20 seconds
            ping_timeout=20   # Wait 20 seconds for pong response
        )
        logging.info(f"WebSocket connection established to {uri}")

        # Start Audio Processing Threads
        sound_level = 20 # Example sound level threshold
        audio_processor_threads = start_audio_processing_threads(
            ap, thread_stop_event, audio_from_processor_queue, sound_level
        )
        if not audio_processor_threads:
            raise RuntimeError("Failed to start essential audio processing threads.")

        # Start TTS Playback Thread
        tts_playback_thread = threading.Thread(
            target=play_tts_audio_thread,
            args=(tts_audio_queue, thread_stop_event),
            name="TTSPlayer",
            daemon=True
        )
        tts_playback_thread.start()
        logging.info("TTS playback thread started.")

        # Create Asyncio Tasks
        receiver_task = asyncio.create_task(
            receive_messages_task(websocket_connection, tts_audio_queue, master_shutdown_event),
            name="ReceiverTask"
        )
        sender_task = asyncio.create_task(
            send_audio_task(websocket_connection, audio_to_send_queue, master_shutdown_event),
            name="SenderTask"
        )
        bridge_task = asyncio.create_task(
            bridge_queue_task(audio_from_processor_queue, audio_to_send_queue, master_shutdown_event),
            name="QueueBridgeTask"
        )
        tasks = [receiver_task, sender_task, bridge_task]

        # Keep the client running until shutdown is signaled
        await master_shutdown_event.wait()
        logging.info("Master shutdown event received.")

    except (websockets.exceptions.InvalidURI, websockets.exceptions.WebSocketException, ConnectionRefusedError, OSError) as e:
        logging.error(f"WebSocket connection failed: {e}", exc_info=True)
    except RuntimeError as e:
         logging.error(f"Runtime error during setup: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred in run_client: {e}", exc_info=True)
        master_shutdown_event.set() # Ensure shutdown is triggered on unexpected errors
    finally:
        logging.info("Starting shutdown sequence...")
        # 1. Signal all tasks and threads to stop
        master_shutdown_event.set() # Signal async tasks
        thread_stop_event.set()     # Signal threads

        # 2. Send finish command if websocket is still open
        if websocket_connection and websocket_connection.open:
            try:
                logging.info("Sending 'finish' command to server...")
                await websocket_connection.send(json.dumps({"command": "finish"}))
                # Give server a moment to process final command (optional)
                await asyncio.sleep(0.2)
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket already closed when trying to send 'finish'.")
            except Exception as e:
                logging.error(f"Error sending 'finish' command: {e}")

        # 3. Cancel running asyncio tasks
        logging.info(f"Cancelling {len(tasks)} async tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to finish cancellation
        await asyncio.gather(*tasks, return_exceptions=True)
        logging.info("Async tasks cancellation complete.")

        # 4. Wait for threads to finish
        all_threads = audio_processor_threads + ([tts_playback_thread] if tts_playback_thread else [])
        logging.info(f"Joining {len(all_threads)} threads...")
        for t in all_threads:
            if t and t.is_alive():
                t.join(timeout=THREAD_JOIN_TIMEOUT)
                if t.is_alive():
                    logging.warning(f"Thread '{t.name}' did not finish within timeout.")
        logging.info("Thread joining complete.")

        # 5. Close WebSocket connection
        if websocket_connection and websocket_connection.open:
            logging.info("Closing WebSocket connection...")
            await websocket_connection.close()
            logging.info("WebSocket connection closed.")

        # 6. Clean up sounddevice (optional, might help release resources)
        try:
            sd._terminate() # Use internal terminate if needed, or just let process end
            logging.info("Sounddevice terminated.")
        except Exception as e:
            logging.warning(f"Error terminating sounddevice: {e}")


        logging.info("Client shutdown complete.")

# --- Entry Point ---

def list_audio_devices():
    """Lists available audio input devices."""
    print("Available audio input devices:")
    try:
        devices = sd.query_devices()
        found_input = False
        for i, device in enumerate(devices):
            # List devices with input channels
            if device.get('max_input_channels', 0) > 0:
                print(f"  [{i}] {device.get('name', 'Unknown Device')} (Input Channels: {device.get('max_input_channels')})")
                found_input = True
            # Optionally list output devices too
            # if device.get('max_output_channels', 0) > 0:
            #     print(f"  [{i}] {device.get('name', 'Unknown Device')} (Output Channels: {device.get('max_output_channels')})")
        if not found_input:
             print("  No input devices found.")
    except Exception as e:
        print(f"  Error querying devices: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="WebSocket Audio Client: Streams microphone audio to a server and plays back TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument("--server", type=str, default="ws://localhost:6789",
                        help="WebSocket server URI (e.g., ws://your_server_ip:port)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit.")
    parser.add_argument("--device", type=int, default=17, # Default to None to let sounddevice choose default
                        help="Input device ID (optional). See --list-devices.")
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help="Sample rate for audio capture.")
device index 17 = CABLE Output (VB-Audio Virtual Cable)
即便在我聽不到聲音的時候，這個裝置仍然會有音量輸出，并且聲音等級能達到100以上，導致不停發送數據到服務器給 asr 識別
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    try:
        asyncio.run(run_client(args.server, args.device, args.rate))
    except KeyboardInterrupt:
        logging.info("Program terminated by user (KeyboardInterrupt).")
    except Exception as e:
        # Catch any unexpected errors during asyncio.run or initial setup
        logging.critical(f"Critical error during client execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
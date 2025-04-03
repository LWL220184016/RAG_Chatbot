import pyaudio
import sounddevice as sd
import multiprocessing
import torch
import time
import numpy as np

from WebSocket.websocket import run_ws_server
from func import asr_process_func, asr_process_func_ws, llm_process_func_ws, tts_process_func

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687" export NEO4J_USERNAME="username" export NEO4J_PASSWORD="password"
# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

ws_host = "localhost"
ws_port = 6789

SOUND_LEVEL = 10
CHUNK = 4096
CHANNELS = 1
RATE = 16000
TIMEOUT_SEC = 0.3

pyaudio_format = pyaudio.paFloat32
process_audio_format = np.float32

is_asr_ready_event = multiprocessing.Event()
is_llm_ready_event = multiprocessing.Event()
is_tts_ready_event = multiprocessing.Event()

stop_event = multiprocessing.Event()
is_user_talking = multiprocessing.Event()
speaking_event = multiprocessing.Event()

client_audio_queue = multiprocessing.Queue()
asr_output_queue = multiprocessing.Queue()
asr_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the user said
llm_output_queue = multiprocessing.Queue()
llm_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the llm said
audio_queue = multiprocessing.Queue()

# asr_output_queue for user text input
ws_process = multiprocessing.Process(
    target=run_ws_server, 
    args=(
        ws_host, 
        ws_port, 
        is_asr_ready_event, 
        is_llm_ready_event, 
        is_tts_ready_event, 
        is_user_talking, 
        client_audio_queue, 
        asr_output_queue, 
        asr_output_queue_ws, # for send back the text to user to show what the user said
        llm_output_queue_ws, 
        audio_queue, 
    )
)
asr_process = multiprocessing.Process( 
    target=asr_process_func, 
    args=( 
        is_user_talking, 
        stop_event, 
        is_asr_ready_event, 
        asr_output_queue, 
        "NeMo", # asr_class: "faster_whisper", "NeMo"
        None, # ap: Audio_Processor
        True, # streaming: True, False
    ) 
) 
asr_process_ws = multiprocessing.Process(
    target=asr_process_func_ws, 
    args=(
        is_user_talking, 
        stop_event, 
        is_asr_ready_event, 
        client_audio_queue, 
        asr_output_queue, 
        asr_output_queue_ws, 
        "NeMo", 
        None, # ap
        True, # streaming: False
    )
)
llm_process_ws = multiprocessing.Process(
    # target=llm_model_process_func_ws, 
    target=llm_process_func_ws, 
    args=( 
        is_user_talking, 
        stop_event, 
        speaking_event, 
        is_llm_ready_event, 
        asr_output_queue, 
        llm_output_queue, 
        llm_output_queue_ws, 
        "google", # llm_class: "google", "transformers"
        True, # use_agent: True, False
        None, # use_database: None, "qdrant"
    )
)
tts_process = multiprocessing.Process(
    target=tts_process_func, 
    args=(
        stop_event, 
        speaking_event, 
        is_tts_ready_event, 
        llm_output_queue, 
        audio_queue, 
    )
)

# Function to initialize and start all processes
def start_all_processes_for_ws():
    # Reset events
    stop_event.clear()
    is_user_talking.clear()
    speaking_event.clear()
    
    # Start all processes
    ws_process.start()
    asr_process_ws.start()
    llm_process_ws.start()
    tts_process.start()
    
    # Wait for all components to be ready
    while not (is_asr_ready_event.is_set() and 
               is_llm_ready_event.is_set() and 
               is_tts_ready_event.is_set()):
        time.sleep(0.1)
    
    print("All processes are ready")

# Function to clean up and terminate processes
def cleanup_processes():
    print("Cleaning up processes...")
    stop_event.set()
    
    for process in [ws_process, asr_process, asr_process_ws, llm_process_ws, tts_process]:
        if process.is_alive():
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
        process.close()
    
    # Clear CUDA cache
    torch.cuda.ipc_collect()
    print("All processes terminated")

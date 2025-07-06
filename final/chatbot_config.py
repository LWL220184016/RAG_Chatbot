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

class Chatbot_config:
    # Set the environment variables for Neo4j and Qdrant
    def __init__(self):
        self.ws_host = "localhost"
        self.ws_port = 6789

        # SOUND_LEVEL = 10
        # CHUNK = 4096
        # CHANNELS = 1
        # RATE = 16000
        # TIMEOUT_SEC = 0.3

        # pyaudio_format = pyaudio.paFloat32
        # process_audio_format = np.float32

        self.is_asr_ready_event = multiprocessing.Event()
        self.is_llm_ready_event = multiprocessing.Event()
        self.is_tts_ready_event = multiprocessing.Event()

        self.stop_event = multiprocessing.Event()
        self.is_user_talking = multiprocessing.Event()
        self.speaking_event = multiprocessing.Event()

        self.client_audio_queue = multiprocessing.Queue()
        self.asr_output_queue = multiprocessing.Queue() # for user text input, or asr output text for llm
        self.asr_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the user said
        self.llm_output_queue = multiprocessing.Queue() # llm output text for tts
        self.llm_output_queue_ws = multiprocessing.Queue() # for send back the text to user to show what the llm said
        self.audio_queue = multiprocessing.Queue() # for tts output audio, then send to websocket

        # self.asr_output_queue for user text input
        self.ws_process = multiprocessing.Process(
            target=run_ws_server, 
            args=(
                self.ws_host, 
                self.ws_port, 
                self.is_asr_ready_event, 
                self.is_llm_ready_event, 
                self.is_tts_ready_event, 
                self.is_user_talking, 
                self.client_audio_queue, 
                self.asr_output_queue, 
                self.asr_output_queue_ws, # for send back the text to user to show what the user said
                self.llm_output_queue_ws, 
                self.audio_queue, 
            )
        )
        self.asr_process = multiprocessing.Process( 
            target=asr_process_func, 
            args=( 
                self.is_user_talking, 
                self.stop_event, 
                self.is_asr_ready_event, 
                self.asr_output_queue, 
                "transformers", # asr_class: "NeMo", "transformers"
                None, # ap: Audio_Processor
                True, # streaming: True, False
            ) 
        ) 
        self.asr_process_ws = multiprocessing.Process(
            target=asr_process_func_ws, 
            args=(
                self.is_user_talking, 
                self.stop_event, 
                self.is_asr_ready_event, 
                self.client_audio_queue, 
                self.asr_output_queue, 
                self.asr_output_queue_ws, 
                "NeMo", # asr_class: "NeMo", "transformers"
                None, # ap: Audio_Processor
                True, # streaming: True, False
            )
        )
        self.llm_process_ws = multiprocessing.Process(
            # target=llm_model_process_func_ws, 
            target=llm_process_func_ws, 
            args=( 
                self.is_user_talking, 
                self.stop_event, 
                self.speaking_event, 
                self.is_llm_ready_event, 
                self.asr_output_queue, 
                self.llm_output_queue, 
                self.llm_output_queue_ws, 
                "transformers", # llm_class: "google", "transformers"
                False, # use_agent: True, False
                None, # use_database: None, "qdrant"
            )
        )
        self.tts_process = multiprocessing.Process(
            target=tts_process_func, 
            args=(
                self.stop_event, 
                self.speaking_event, 
                self.is_tts_ready_event, 
                self.llm_output_queue, 
                self.audio_queue, 
            )
        )

    # Function to initialize and start all processes
    def start_all_processes_for_ws(self):
        # Reset events
        self.stop_event.clear()
        self.is_user_talking.clear()
        self.speaking_event.clear()
        
        # Start all processes
        self.ws_process.start()
        self.asr_process_ws.start()
        self.llm_process_ws.start()
        self.tts_process.start()
        
        # Wait for all components to be ready
        while not (self.is_asr_ready_event.is_set() and 
                self.is_llm_ready_event.is_set() and 
                self.is_tts_ready_event.is_set()):
            time.sleep(0.1)
        
        print("All processes are ready")

    # Function to clean up and terminate processes
    def cleanup_processes(self):
        print("Cleaning up processes...")
        self.stop_event.set()
        
        for process in [self.ws_process, self.asr_process, self.asr_process_ws, self.llm_process_ws, self.tts_process]:
            if process.is_alive():
                process.join(timeout=3.0)
                if process.is_alive():
                    process.terminate()
            process.close()
        
        # Clear CUDA cache
        torch.cuda.ipc_collect()
        print("All processes terminated")

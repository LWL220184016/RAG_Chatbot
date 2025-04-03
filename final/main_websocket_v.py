import pyaudio
import sounddevice as sd
import torch
import time

from veriables import (
    stop_event, 
    is_user_talking, 
    speaking_event, 
    audio_queue, 
    start_all_processes_for_ws, 
    cleanup_processes, 
)

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687" export NEO4J_USERNAME="username" export NEO4J_PASSWORD="password"
# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    try:
        start_all_processes_for_ws()
            
        while not stop_event.is_set():
            while speaking_event.is_set() or not audio_queue.empty():
                audio_chunk = audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=False)
                while sd.get_stream().active:
                    if is_user_talking.is_set():
                        sd.stop()
                        if not audio_queue.empty():
                            audio_chunk = audio_queue.get()
                        break
                    time.sleep(0.01)


            time.sleep(1)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        cleanup_processes()
        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
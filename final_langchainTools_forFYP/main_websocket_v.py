import pyaudio
import sounddevice as sd
import torch
import time

from chatbot_config import Chatbot_config

# set environment variable in linux
# export NEO4J_URI="neo4j://localhost:7687" export NEO4J_USERNAME="username" export NEO4J_PASSWORD="password"
# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    try:
        chatbot_config = Chatbot_config()
        chatbot_config.start_all_processes_for_ws()
            
        while not chatbot_config.stop_event.is_set():
            while chatbot_config.speaking_event.is_set() or not chatbot_config.audio_queue.empty():
                audio_chunk = chatbot_config.audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=False)
                while sd.get_stream().active:
                    if chatbot_config.is_user_talking.is_set():
                        sd.stop()
                        if not chatbot_config.audio_queue.empty():
                            audio_chunk = chatbot_config.audio_queue.get()
                        break
                    time.sleep(0.01)


            time.sleep(1)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        chatbot_config.cleanup_processes()
        torch.cuda.ipc_collect()
        print("User stopped the program\n")

if __name__ == "__main__":
    main()
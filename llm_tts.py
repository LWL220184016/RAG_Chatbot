import threading
import queue
import time
from langchain_ollama import OllamaLLM
import base64
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset 
import torch 
import sounddevice as sd 
import numpy as np

def get_LLM():
    # Initialize langchain ollama with GGUF format model
    langchain_llm = OllamaLLM(
        # model="Qwen2-VL-7B-Instruct",
        model="llama3.2-vision",
        # model="llama3.3:70b-instruct-q2_K",
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    return langchain_llm

def llm_output(llm, prompt_queue, output_queue):
    while True:
        prompt = prompt_queue.get()
        text = ""
        for output in llm.invoke(prompt):
            if output not in ["，", ",", "。", ".", "？", "?", "！", "!"]:
                text += output
            else:
                output_queue.put(text)
                text = ""
        prompt_queue.task_done()

def tts_output(processor, model, vocoder, output_queue, audio_queue, speaking_event):
    while True:
        if output_queue.empty():
            speaking_event.clear()  # Signal that LLM has finished speaking

        text = output_queue.get()
        while audio_queue.qsize() >= 5:
            pass
        print("LLM: ", text)
        inputs = processor(text=text, return_tensors="pt") 
        audio_chunk = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        audio_chunk = audio_chunk.cpu().numpy()
        audio_queue.put(audio_chunk)
        speaking_event.set()  # Signal that LLM is speaking
        output_queue.task_done()

# Load the processor, model, and vocoder 
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") 
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts") 
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan") 

# Load speaker embeddings 
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") 
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) 

if __name__ == "__main__":
    llm = get_LLM()

    prompt_queue = queue.Queue()
    output_queue = queue.Queue()
    audio_queue = queue.Queue()
    speaking_event = threading.Event()

    llm_thread = threading.Thread(target=llm_output, args=(llm, prompt_queue, output_queue))
    tts_thread = threading.Thread(target=tts_output, args=(processor, model, vocoder, output_queue, audio_queue, speaking_event))

    llm_thread.start()
    tts_thread.start()

    # for audio_data in record_and_detect_vad():

    try:
        while True:
            prompt = input("You: ")
            prompt_queue.put(prompt)
            speaking_event.set()  # Signal that LLM is speaking
            
            # while not audio_queue.empty():
            while speaking_event.is_set() or not audio_queue.empty():
                audio_chunk = audio_queue.get()
                sd.play(audio_chunk, samplerate=16000, blocking=True)
                sd.wait()
                audio_queue.task_done()

    except KeyboardInterrupt:
        pass
    finally:
        prompt_queue.put(None)
        output_queue.put(None)
        llm_thread.join()
        tts_thread.join()


# Traceback (most recent call last):
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_transports\default.py", line 72, in map_httpcore_exceptions
#     yield
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_transports\default.py", line 236, in handle_request
#     resp = self._pool.handle_request(req)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_sync\connection_pool.py", line 256, in handle_request
#     raise exc from None
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_sync\connection_pool.py", line 236, in handle_request
#     response = connection.handle_request(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_sync\connection.py", line 101, in handle_request
#     raise exc
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_sync\connection.py", line 78, in handle_request
#     stream = self._connect(request)
#              ^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_sync\connection.py", line 124, in _connect
#     stream = self._network_backend.connect_tcp(**kwargs)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_backends\sync.py", line 207, in connect_tcp
#     with map_exceptions(exc_map):
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\contextlib.py", line 158, in __exit__
#     self.gen.throw(value)
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpcore\_exceptions.py", line 14, in map_exceptions
#     raise to_exc(exc) from exc
# httpcore.ConnectError: [WinError 10061] 由于目标计算机积极拒绝，无法连接。

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\threading.py", line 1073, in _bootstrap_inner
#     self.run()
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\threading.py", line 1010, in run
#     self._target(*self._args, **self._kwargs)
#   File "c:\Users\User\Desktop\AI\RAG_Chatbot\llm_tts.py", line 28, in llm_output
#     for output in llm.invoke(prompt):
#                   ^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_core\language_models\llms.py", line 390, in invoke
#     self.generate_prompt(
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_core\language_models\llms.py", line 755, in generate_prompt
#     return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_core\language_models\llms.py", line 950, in generate
#     output = self._generate_helper(
#              ^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_core\language_models\llms.py", line 792, in _generate_helper
#     raise e
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_core\language_models\llms.py", line 779, in _generate_helper
#     self._generate(
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_ollama\llms.py", line 288, in _generate
#     final_chunk = self._stream_with_aggregation(
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_ollama\llms.py", line 256, in _stream_with_aggregation
#     for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\langchain_ollama\llms.py", line 211, in _create_generate_stream
#     yield from self._client.generate(
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\ollama\_client.py", line 162, in inner
#     with self._client.stream(*args, **kwargs) as r:
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\contextlib.py", line 137, in __enter__
#     return next(self.gen)
#            ^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_client.py", line 880, in stream
#     response = self.send(
#                ^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_client.py", line 926, in send
#     response = self._send_handling_auth(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_client.py", line 954, in _send_handling_auth
#     response = self._send_handling_redirects(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_client.py", line 991, in _send_handling_redirects
#     response = self._send_single_request(request)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_client.py", line 1027, in _send_single_request
#     response = transport.handle_request(request)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_transports\default.py", line 235, in handle_request
#     with map_httpcore_exceptions():
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\contextlib.py", line 158, in __exit__
#     self.gen.throw(value)
#   File "C:\Users\User\anaconda3\envs\py3.12\Lib\site-packages\httpx\_transports\default.py", line 89, in map_httpcore_exceptions
#     raise mapped_exc(message) from exc
# httpx.ConnectError: [WinError 10061] 由于目标计算机积极拒绝，无法连接。
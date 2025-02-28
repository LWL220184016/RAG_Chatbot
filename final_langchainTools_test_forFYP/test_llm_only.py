import multiprocessing
import torch
import time

# from LLM.llm_ollama import LLM_Ollama as LLM
from LLM.llm_google import LLM_Google as LLM
# from LLM.llm_transformers import LLM_Transformers as LLM
from LLM.prompt_template import get_langchain_PromptTemplate_Chinese2
from func_fyp import llm_model_process_func_ws, llm_agent_process_func_ws, llm_memory_model_process_func_ws, llm_memory_agent_process_func_ws
from langchain_community.agent_toolkits.load_tools import load_tools


# from Tools.duckduckgo_searching import duckduckgo_search
# from Tools.querying_qdrant import  querying_qdrant
from Tools.tool import Tools

from Data_Storage.qdrant import Qdrant_Handler as Database
from Data_Storage.embedding_model.embedder import Embedder
# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

def main():
    
    is_llm_ready_event = multiprocessing.Event()

    stop_event = multiprocessing.Event()
    is_user_talking = multiprocessing.Event()
    speaking_event = multiprocessing.Event()

    asr_output_queue = multiprocessing.Queue()
    llm_output_queue = multiprocessing.Queue()
    llm_output_queue_ws = multiprocessing.Queue()

    embedder = Embedder()
    database = Database(embedder=embedder) # 嘗試吧 embedder model 塞到另一個進程或者線程來解決 RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    tools = Tools(database_qdrant=database)
    tools=[tools.duckduckgo_search, tools.querying_qdrant]
    llm = LLM( 
        # model_name="deepseek-r1_14b_FYP4", 
        # torch_dtype=torch.float32, 
        # device="cuda:0", 
        is_user_talking=is_user_talking, 
        stop_event=stop_event, 
        speaking_event=speaking_event, 
        user_input_queue=asr_output_queue, 
        llm_output_queue=llm_output_queue, 
        llm_output_queue_ws=llm_output_queue_ws, 
        tools=tools, 
        database=database, 
    ) 


problem useing search qdrant tool, error:
Traceback (most recent call last):
  File "/workspaces/RAG_Chatbot/final_langchainTools_test_forFYP/func_fyp.py", line 124, in llm_agent_process_func_ws
    llm.agent_output_ws(
  File "/workspaces/RAG_Chatbot/final_langchainTools_test_forFYP/LLM/llm_google.py", line 84, in agent_output_ws
    super().agent_output_ws(self.agent, is_llm_ready_event, prompt_template)
  File "/workspaces/RAG_Chatbot/final_langchainTools_test_forFYP/LLM/llm.py", line 66, in agent_output_ws
    agent.invoke(user_input)
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain/chains/base.py", line 170, in invoke
    raise e
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain/chains/base.py", line 160, in invoke
    self._call(inputs, run_manager=run_manager)
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain/agents/agent.py", line 1624, in _call
    next_step_output = self._take_next_step(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain/agents/agent.py", line 1330, in _take_next_step
    [
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain/agents/agent.py", line 1415, in _iter_next_step
    yield self._perform_agent_action(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain/agents/agent.py", line 1437, in _perform_agent_action
    observation = tool.run(
                  ^^^^^^^^^
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain_core/tools/base.py", line 725, in run
    raise error_to_raise
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain_core/tools/base.py", line 689, in run
    tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain_core/tools/base.py", line 611, in _to_args_and_kwargs
    tool_input = self._parse_input(tool_input, tool_call_id)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/langchain_core/tools/base.py", line 532, in _parse_input
    result = input_args.model_validate(tool_input)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python/3.12.1/lib/python3.12/site-packages/pydantic/main.py", line 596, in model_validate
    return cls.__pydantic_validator__.validate_python(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for querying_qdrant
self
  Field required [type=missing, input_value={'query': 'user question'...n_name': 'chat_2024_10'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.9/v/missing

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/python/3.12.1/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python/3.12.1/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/workspaces/RAG_Chatbot/final_langchainTools_test_forFYP/func_fyp.py", line 134, in llm_agent_process_func_ws
    torch.cuda.ipc_collect()
  File "/home/codespace/.local/lib/python3.12/site-packages/torch/cuda/__init__.py", line 966, in ipc_collect
    _lazy_init()
  File "/home/codespace/.local/lib/python3.12/site-packages/torch/cuda/__init__.py", line 310, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled



    prompt_template = get_langchain_PromptTemplate_Chinese2()

    try:
        llm_process = multiprocessing.Process(
            # target=llm_model_process_func_ws, 
            target=llm_agent_process_func_ws, 
            # target=llm_memory_model_process_func_ws, 
            # target=llm_memory_agent_process_func_ws, 
            args=(
                stop_event, 
                is_user_talking, 
                speaking_event, 
                is_llm_ready_event, 
                asr_output_queue, 
                llm_output_queue, 
                llm_output_queue_ws,
                prompt_template,
                llm,
            )
        )
        
        llm_process.start()


        asr_output_queue.put("Hello")

        while not is_llm_ready_event.is_set():
            time.sleep(0.1)

        while not stop_event.is_set():
            user_input = input()
            if user_input == "show":
                while not llm_output_queue.empty():
                    output = llm_output_queue.get_nowait()
                    print(f"\n\033[38;5;208m🔍 LLM: {output}\033[0m")  # 橙色高亮 (256-color)
            else:
                asr_output_queue.put(user_input)
            
    except KeyboardInterrupt:
        print("main KeyboardInterrupt\n")
        stop_event.set()
        llm_process.join()
        llm_process.close()
        
        torch.cuda.ipc_collect()
        print("User stopped the program\n")


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main()
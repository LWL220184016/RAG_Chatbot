import queue
import multiprocessing
import time
import threading

from langchain_ollama import OllamaLLM
from LLM.prompt_template import Message
# from Data_Storage.neo4j import Neo4J

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import Tool
from tenacity import retry, stop_after_attempt, wait_fixed

from Tools.duckduckgo_searching import duckduckgo_search
from LLM.llmAgentStreamingCallbackHandler import OllamaAgentStreamingCallbackHandler

class LLM:
    def __init__(
            self,
            model_name: str = "deepseek-r1:14b",
            top_k: int = 10,
            top_p: float = 0.95,
            temperature: float = 0.8,
            is_user_talking = None,
            stop_event = None,
            speaking_event = None,
            user_input_queue: multiprocessing.Queue = None,
            llm_output_queue: multiprocessing.Queue = None,
            llm_output_queue_ws: multiprocessing.Queue = None,
            tools = [],
            # neo4j: Neo4J = Neo4J()
        ):

        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        self.is_user_talking = is_user_talking 
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")
        # self.neo4j = neo4j

        # self.callback_queue = multiprocessing.Queue()
        custom_callback = OllamaAgentStreamingCallbackHandler(
            is_user_talking=self.is_user_talking, 
            user_input_queue=self.user_input_queue, 
            llm_output_queue=self.llm_output_queue,
            llm_output_queue_ws=self.llm_output_queue_ws,
        )
        self.model = OllamaLLM(
            model=model_name,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            callbacks=[StreamingStdOutCallbackHandler(), custom_callback],  # 标准输出回调
        )
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors="Check your output format!",
            callbacks=[custom_callback],  # 绑定自定义回调
        )

    def llm_output_ws(
            self,
            prompt_template = None,
            rag=None
        ):

        user_input = ""
        user_last_talk_time = time.time()

        try:
            while not self.stop_event.is_set():
                if not self.is_user_talking.is_set():
                    if time.time() - user_last_talk_time > 5:
                        user_input = ""
                    try:
                        user_input += self.user_input_queue.get(timeout=0.1) + " "
                    except queue.Empty:
                        continue
                    if not self.user_input_queue.empty():
                        user_input += self.user_input_queue.get() + " "
                else: # user is talking
                    user_last_talk_time = time.time()
                    continue  # Skip if the user is talking

                print("user_input: " + user_input + "  -----------------------------------------------------user_input")
                
                # Assuming 'rag' has a 'search' method that takes 'llm' and 'query' as parameters
                prompt = "return the previous dialogue content relate to the queue"
                # memory = rag.search_rag(query=user_input, prompt=prompt, mode="hybrid")
            
# have a problem with the rag
                self.speaking_event.set()
                self.agent.invoke(prompt_template.format(user_input=user_input))
        except KeyboardInterrupt:
            print("llm_output_ws KeyboardInterrupt\n")
            self.stop_event.set()
            
            # torch.cuda.ipc_collect()
            print("User stopped the program\n")

#     def check_llm_output(
#             self, 
#             user_input_queue: multiprocessing.Queue, 
#         ):
#         llm_output = ""
#         llm_output_total = ""
#         is_llm_thinking = False

#         while not self.stop_event.is_set():
#             try:
#                 output = self.callback_queue.get(timeout=0.1)
#             except queue.Empty:
#                 continue
#             if self.is_user_talking.is_set() or not user_input_queue.empty():
#                 if not self.llm_output_queue.empty():
#                     empty_queue = self.llm_output_queue.get(block=False)
#                     empty_queue = self.callback_queue.get(block=False)
#                 break

#             if output == None:
#                 llm_output = ""
#                 llm_output_total = ""
#                 is_llm_thinking = False
#                 continue
            
#             # Directly append to llm_output, reducing queue operations
#             llm_output += output
#             if "<think>" in output and not is_llm_thinking:
#                 is_llm_thinking = True
#                 print("is_llm_thinking = True")
#             elif "</think>" in output and is_llm_thinking:
#                 is_llm_thinking = False
#                 print("is_llm_thinking = False")
#             if output in ["，", ",", "。", ".", "？", "?", "！", "!"] or "</think>" in output:
#                 llm_output_total += llm_output
#                 print("\n\n   ---llm output: " + llm_output + "---\n\n")
#                 if not is_llm_thinking and "</think>" in output:
#                     self.llm_output_queue.put(llm_output)
#                 self.llm_output_queue_ws.put(llm_output)
#                 llm_output = ""


#             # self.neo4j.add_dialogue_record(user_message, llm_message)
#             llm_output_total = ""
# # 尝试在 OllamaAgentStreamingCallbackHandler 的 on_llm_new_token 方法中添加侦测，判断 action 是否 Final Answer
# # 如果是 Final Answer，就将其放入 llm_output_queue 中，否则放入 llm_output_queue_ws 中
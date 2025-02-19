import multiprocessing
import time
import queue

from langchain_google_genai import ChatGoogleGenerativeAI
# from Data_Storage.neo4j import Neo4J
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import Tool
from tenacity import retry, stop_after_attempt, wait_fixed

from Tools.duckduckgo_searching import duckduckgo_search
from LLM.llmAgentStreamingCallbackHandler import GoogleAgentStreamingCallbackHandler

class LLM:
    def __init__(
            self,
            model_name: str = "gemini-1.5-flash",
            temperature: float = 0.0,
            max_tokens: int = None,
            timeout: float = None,
            max_retries: int = 2,
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
        
        custom_callback = GoogleAgentStreamingCallbackHandler(
            is_user_talking=self.is_user_talking, 
            user_input_queue=self.user_input_queue, 
            llm_output_queue=self.llm_output_queue, 
            llm_output_queue_ws=self.llm_output_queue_ws, 
        )
        self.model = ChatGoogleGenerativeAI(
            # model="gemini-1.5-pro",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            streaming=True,  # 启用流式传输
            callbacks=[StreamingStdOutCallbackHandler(), custom_callback],  # 标准输出回调
            # other params...
        )
        self.agent = initialize_agent(
            tools=tools,
            llm=self.model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors="Check your output format!",
            callbacks=[custom_callback],  # 绑定自定义回调
        )

    def agent_output_ws(
            self,
            prompt_template = None,
            rag=None
        ):

        user_input = ""
        user_last_talk_time = time.time()
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
            

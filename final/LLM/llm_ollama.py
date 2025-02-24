import multiprocessing

from llm import LLM
from langchain_ollama import OllamaLLM
# from Data_Storage.neo4j import Neo4J
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType, initialize_agent
from tenacity import retry, stop_after_attempt, wait_fixed
from prompt_template import Message

class LLM_Ollama(LLM):
    def __init__(
            self, 
            top_k: int = 10, 
            top_p: float = 0.95, 
            temperature: float = 0.8, 

            model_name: str = "deepseek-r1:14b", 
            is_user_talking = None, 
            stop_event = None, 
            speaking_event = None, 
            user_input_queue: multiprocessing.Queue = None, 
            llm_output_queue: multiprocessing.Queue = None, 
            llm_output_queue_ws: multiprocessing.Queue = None, 
            tools = [], 
            # neo4j: Neo4J = Neo4J()
        ):
        
        super().__init__(
            model_name, 
            is_user_talking, 
            stop_event, 
            speaking_event, 
            user_input_queue, 
            llm_output_queue, 
            llm_output_queue_ws, 
            tools, 
        )

        self.is_user_talking = is_user_talking 
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        # self.neo4j = neo4j

        # custom_callback = OllamaAgentStreamingCallbackHandler(
        #     is_user_talking=self.is_user_talking, 
        #     user_input_queue=self.user_input_queue, 
        #     llm_output_queue=self.llm_output_queue,
        #     llm_output_queue_ws=self.llm_output_queue_ws,
        # )
        self.model = OllamaLLM(
            model=model_name,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            # callbacks=[StreamingStdOutCallbackHandler(), custom_callback],  # 标准输出回调
            callbacks=[StreamingStdOutCallbackHandler()],  # 标准输出回调
        )
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors="Check your output format!",
            # callbacks=[custom_callback],  # 绑定自定义回调
        )

    def agent_output_ws(
            self,
            is_llm_ready_event, 
            user_message: Message = None,
            llm_message: Message = None,
            rag=None
        ):

        # 在這裡傳遞必要的參數給父類別的方法
        super().agent_output_ws(self.agent, is_llm_ready_event, user_message, llm_message, rag)

    def llm_output_ws(
            self, 
            is_llm_ready_event, 
            user_message: Message = None,
            llm_message: Message = None,
            rag=None, 
        ):
        
        # 在這裡傳遞必要的參數給父類別的方法
        super().llm_output_ws(self.model, is_llm_ready_event, user_message, llm_message, rag)
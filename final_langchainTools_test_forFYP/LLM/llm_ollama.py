import queue

from LLM.llm import LLM
from LLM.llmAgentStreamingCallbackHandler import OllamaAgentStreamingCallbackHandler
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType, initialize_agent
from tenacity import retry, stop_after_attempt, wait_fixed

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
            user_input_queue: queue = None, 
            llm_output_queue: queue = None, 
            llm_output_queue_ws: queue = None, 
            tools = [], 
            database = None,
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
            database, 
        )

        self.is_user_talking = is_user_talking 
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws

        custom_callback = OllamaAgentStreamingCallbackHandler(
            is_user_talking=self.is_user_talking, 
            user_input_queue=self.user_input_queue, 
            llm_output_queue=self.llm_output_queue,
            llm_output_queue_ws=self.llm_output_queue_ws,
            database=database,
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

    def agent_output_ws(
            self,
            is_llm_ready_event, 
            prompt_template = None,
        ):

        # 在這裡傳遞必要的參數給父類別的方法
        super().agent_output_ws(self.agent, is_llm_ready_event, prompt_template)

    def llm_output_ws(
            self, 
            is_llm_ready_event, 
            prompt_template = None, 
        ):
        
        # 在這裡傳遞必要的參數給父類別的方法
        super().llm_output_ws(self.model, is_llm_ready_event, prompt_template)

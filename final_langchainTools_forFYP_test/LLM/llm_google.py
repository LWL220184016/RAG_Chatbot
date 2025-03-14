import queue

from LLM.llm import LLM
from LLM.llmAgentStreamingCallbackHandler import GoogleAgentStreamingCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType, initialize_agent, AgentExecutor

from tenacity import retry, stop_after_attempt, wait_fixed

class LLM_Google(LLM):
    def __init__(
            self, 
            temperature: float = 0.0, 
            max_tokens: int = None, 
            timeout: float = None, 
            max_retries: int = 2, 

            model_name: str = "gemini-1.5-flash", 
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
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None, they should be multiprocessing.Event() objects.")
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        
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
        
        raise NotImplementedError("llm_output_ws() in llm_google is not implemented yet.")

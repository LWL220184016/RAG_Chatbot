from langchain_ollama import OllamaLLM
import queue
import threading
from LLM.prompt_template import Message

class LLM():
    def __init__(
            self, 
            model_name: str = "llama3.2-vision-latest-friend2", 
            top_k: int = 10, 
            top_p: float = 0.95, 
            temperature: float = 0.8, 
            is_user_talking: threading.Event = None,
            stop_event: threading.Event = None,
            speaking_event: threading.Event = None
        ):

        self.model = OllamaLLM(
            model=model_name,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        self.llm_output_queue = queue.Queue()
        self.is_user_talking = is_user_talking
        self.stop_event = stop_event
        self.speaking_event = speaking_event

    def llm_output(
        self, 
        user_input_queue, 
        user_message: Message = None, 
        llm_message: Message = None, 
        rag=None
    ):
        
        while not self.stop_event.is_set():
            user_input = user_input_queue.get()
            memory = rag.search(llm=self.model, query=user_input)

            msg = user_message.update_content(content=user_input + "; " + memory)
            text_parts = []
            llm_output = ""
            self.speaking_event.set()
            for output in self.model.invoke(msg):
                if output not in ["，", ",", "。", ".", "？", "?", "！", "!"]:
                    text_parts.append(output)
                else:
                    text = ''.join(text_parts)
                    self.llm_output_queue.put(text)
                    llm_output += text
                    text_parts = []
            
            llm_message.update_content(content=user_input_queue.get())
            rag.update_chat_history(user_message, llm_message)
            user_input_queue.task_done()
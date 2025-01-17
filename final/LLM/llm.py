from langchain_ollama import OllamaLLM
import queue
import threading
from LLM.prompt_template import Message

class LLM:
    def __init__(
            self,
            model_name: str = "llama3.2-vision-latest-friend2",
            top_k: int = 10,
            top_p: float = 0.95,
            temperature: float = 0.8,
            is_user_talking: threading.Event = None,
            stop_event: threading.Event = threading.Event(),
            speaking_event: threading.Event = threading.Event()
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
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")

    def llm_output(
            self,
            user_input_queue: queue.Queue,
            user_message: Message = None,
            llm_message: Message = None,
            rag=None
        ):
        
        prev_msg = ""
        while not self.stop_event.is_set():
            try:
                user_input = ""
                if not self.is_user_talking.is_set():
                    try:
                        user_input = user_input_queue.get(timeout=1)  # Wait for 1 second
                    except queue.Empty:
                        pass
                else:
                    continue  # Skip if the user is talking

                self.is_user_talking.clear()
                user_input = prev_msg + " " + user_input
                print("user_input: " + user_input)
                
                # Assuming 'rag' has a 'search' method that takes 'llm' and 'query' as parameters
                memory = rag.search(llm=self.model, query=user_input)
                
                # Assuming 'update_content' method exists for Message class
                msg = user_message.update_content(content=user_input, memory=memory)
                prev_msg = ""

                self.speaking_event.set()
                llm_output = ""
                llm_output_total = ""
                for output in self.model.invoke(msg):
                    if self.is_user_talking.is_set():
                        prev_msg = msg  # Corrected to directly assign msg to prev_msg
                        break
                    
                    # Directly append to llm_output, reducing queue operations
                    llm_output += output
                    if output in ["，", ",", "。", ".", "？", "?", "！", "!"]:
                        llm_output_total += llm_output
                        self.llm_output_queue.put(llm_output)
                        llm_output = ""

                # Assuming 'update_content' method exists for Message class
                llm_message.update_content(content=llm_output_total)
                rag.update_chat_history(user_message, llm_message)
                llm_output_total = ""
                user_input_queue.task_done()
            
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                user_input_queue.task_done()


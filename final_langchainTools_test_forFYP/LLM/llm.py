import queue
import multiprocessing
import time
from langchain_ollama import OllamaLLM
from LLM.prompt_template import Message
# from Data_Storage.neo4j import Neo4J

class LLM:
    def __init__(
            self,
            model_name: str = "deepseek-r1-14b-qwen-fd",
            top_k: int = 10,
            top_p: float = 0.95,
            temperature: float = 0.8,
            is_user_talking = None,
            stop_event = None,
            speaking_event = None,
            llm_output_queue: multiprocessing.Queue = None,
            # neo4j: Neo4J = Neo4J()
        ):

        self.model = OllamaLLM(
            model=model_name,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        self.llm_output_queue = llm_output_queue
        self.is_user_talking = is_user_talking 
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")
        # self.neo4j = neo4j

    def llm_output(
            self,
            user_input_queue: queue.Queue = None,
            user_message: Message = None,
            llm_message: Message = None,
            rag=None
        ):

        user_input = ""
        user_last_talk_time = time.time()
        while not self.stop_event.is_set():
                if not self.is_user_talking.is_set():
                    if time.time() - user_last_talk_time > 5:
                        user_input = ""
                    try:
                        user_input += user_input_queue.get(timeout=0.1) + " "
                    except queue.Empty:
                        continue
                    if not user_input_queue.empty():
                        user_input += user_input_queue.get() + " "
                else: # user is talking
                    user_last_talk_time = time.time()
                    continue  # Skip if the user is talking

                print("user_input: " + user_input + "  -----------------------------------------------------user_input")
                
                # Assuming 'rag' has a 'search' method that takes 'llm' and 'query' as parameters
                prompt = "return the previous dialogue content relate to the queue"
                memory = rag.search_rag(query=user_input, prompt=prompt, mode="hybrid")
                
                # Assuming 'update_content' method exists for Message class
                msg = user_message.update_content(content=user_input, memory=memory)
# have a problem with the rag
                self.speaking_event.set()
                llm_output = ""
                llm_output_total = ""
                for output in self.model.stream(msg):
                    if self.is_user_talking.is_set() or not user_input_queue.empty():
                        if not self.llm_output_queue.empty():
                            empty_queue = self.llm_output_queue.get(block=False)
                        break
                    
                    # Directly append to llm_output, reducing queue operations
                    llm_output += output
                    if output in ["，", ",", "。", ".", "？", "?", "！", "!"]:
                        llm_output_total += llm_output
                        print("llm output: " + llm_output)
                        self.llm_output_queue.put(llm_output)
                        llm_output = ""

                # Assuming 'update_content' method exists for Message class
                llm_message.update_content(content=llm_output_total)
                self.neo4j.add_dialogue_record(user_message, llm_message)
                llm_output_total = ""
                # user_input = ""

    def llm_output_ws(
            self,
            user_input_queue: queue.Queue = None,
            llm_output_queue_ws: queue.Queue = None, 
            user_message: Message = None,
            llm_message: Message = None,
            rag=None
        ):

        user_input = ""
        user_last_talk_time = time.time()
        while not self.stop_event.is_set():
                if not self.is_user_talking.is_set():
                    if time.time() - user_last_talk_time > 5:
                        user_input = ""
                    try:
                        user_input += user_input_queue.get(timeout=0.1) + " "
                    except queue.Empty:
                        continue
                    if not user_input_queue.empty():
                        user_input += user_input_queue.get() + " "
                else: # user is talking
                    user_last_talk_time = time.time()
                    continue  # Skip if the user is talking

                print("user_input: " + user_input + "  -----------------------------------------------------user_input")
                
                # Assuming 'rag' has a 'search' method that takes 'llm' and 'query' as parameters
                prompt = "return the previous dialogue content relate to the queue"
                # memory = rag.search_rag(query=user_input, prompt=prompt, mode="hybrid")
                
                # Assuming 'update_content' method exists for Message class
                # msg = user_message.update_content(content=user_input, memory=memory)
                msg = user_message.update_content(content=user_input, memory=None)
# have a problem with the rag
                self.speaking_event.set()
                llm_output = ""
                llm_output_total = ""
                is_llm_thinking = False
                for output in self.model.stream(msg):
                    if self.is_user_talking.is_set() or not user_input_queue.empty():
                        if not self.llm_output_queue.empty():
                            empty_queue = self.llm_output_queue.get(block=False)
                        break
                    
                    # Directly append to llm_output, reducing queue operations
                    llm_output += output
                    if output == "<think>" and not is_llm_thinking:
                        is_llm_thinking = True
                        print("is_llm_thinking = True")
                    elif output == "</think>" and is_llm_thinking:
                        is_llm_thinking = False
                        print("is_llm_thinking = False")
                    if output in ["，", ",", "。", ".", "？", "?", "！", "!"] or "</think>" in output:
                        llm_output_total += llm_output
                        print("llm output: " + llm_output)
                        if not is_llm_thinking or "</think>" in output:
                            self.llm_output_queue.put(llm_output)
                            print("after put llm_output_queue: ", self.llm_output_queue.qsize())
                        llm_output_queue_ws.put(llm_output)
                        llm_output = ""

                # Assuming 'update_content' method exists for Message class
                llm_message.update_content(content=llm_output_total)
                # self.neo4j.add_dialogue_record(user_message, llm_message)
                llm_output_total = ""

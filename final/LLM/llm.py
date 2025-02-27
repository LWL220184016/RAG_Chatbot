import queue
import multiprocessing
import time

# from Data_Storage.neo4j import Neo4J
from tenacity import retry, stop_after_attempt, wait_fixed
from prompt_template import Message

class LLM:
    def __init__(
            self, 
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

        self.is_user_talking = is_user_talking 
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        # self.neo4j = neo4j

        self.model_name = model_name
        self.tools = tools
    
    def agent_output_ws(
            self,
            agent,
            is_llm_ready_event, 
            user_message: Message = None,
            llm_message: Message = None,
            rag=None
        ):

        print("llm waiting text")
        is_llm_ready_event.set()
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
                prompt = "return the previous chat content relate to the queue"
                # memory = rag.search_rag(query=user_input, prompt=prompt, mode="hybrid")
            
# have a problem with the rag
                self.speaking_event.set()
                # agent.invoke(prompt_template.format(user_input=user_input))
                agent.invoke(user_input.format(user_input=user_input))
        except KeyboardInterrupt:
            print("agent_output_ws KeyboardInterrupt\n")
            self.stop_event.set()
            
            # torch.cuda.ipc_collect()
            print("User stopped the program\n")

    def llm_output_ws(
            self, 
            model,
            is_llm_ready_event, 
            user_message: Message = None,
            llm_message: Message = None,
            rag=None, 
        ):

        print("llm waiting text")
        is_llm_ready_event.set()
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
                prompt = "return the previous chat content relate to the queue"
                # memory = rag.search_rag(query=user_input, prompt=prompt, mode="hybrid")
                
                # Assuming 'update_content' method exists for Message class
                # msg = user_message.update_content(content=user_input, memory=memory)
                # msg = user_message.update_content(content=user_input, memory=None)
# have a problem with the rag
                self.speaking_event.set()
                llm_output = ""
                llm_output_total = ""
                is_llm_thinking = False
                
                # self.agent.invoke(prompt_template.format(user_input=user_input))
                for output in model.stream(user_input):
                    if self.is_user_talking.is_set() or not self.user_input_queue.empty():
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
                        self.llm_output_queue_ws.put(llm_output)
                        llm_output = ""

                # Assuming 'update_content' method exists for Message class
                # llm_message.update_content(content=llm_output_total)
                # self.neo4j.add_chat_record(user_message, llm_message)
                llm_output_total = ""

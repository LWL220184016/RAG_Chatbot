import queue
import multiprocessing
# Commenting out to allow using the default 'fork' start method on Linux
# multiprocessing.set_start_method('spawn', force=False)
import time

from Data_Storage.database import Database_Handler
from tenacity import retry, stop_after_attempt, wait_fixed

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
            database: Database_Handler = None
        ):

        self.is_user_talking = is_user_talking 
        self.stop_event = stop_event
        self.speaking_event = speaking_event
        if self.is_user_talking is None or self.stop_event is None or self.speaking_event is None:
            raise ValueError("is_user_talking, stop_event, and speaking_event must not be None")
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        self.database = database

    
    def agent_output_ws( 
            self, 
            agent = None, 
            is_llm_ready_event = None, 
            prompt_template = None, 
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

                print(f"\033[95mUser: {user_input} \033[0m")  # 紫色高亮输出
                
                self.speaking_event.set()
                # agent.invoke(prompt_template.format(user_input=user_input))
                agent.invoke(user_input)
        except KeyboardInterrupt:
            print("agent_output_ws KeyboardInterrupt\n")
            self.stop_event.set()
            
            # torch.cuda.ipc_collect()
            print("User stopped the program\n")

    def llm_output_ws(
            self, 
            model = None, 
            is_llm_ready_event = None, 
            prompt_template = None, 
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

                print(f"\033[95mUser: {user_input} \033[0m")  # 紫色高亮输出

                self.speaking_event.set()
                llm_output = ""
                llm_output_total = ""
                is_llm_thinking = False

                # for output in model.stream(prompt_template.format(user_input=user_input)):
                for output in model.stream(user_input):
                    if self.is_user_talking.is_set() or not self.user_input_queue.empty():
                        if not self.llm_output_queue.empty():
                            empty_queue = self.llm_output_queue.get(block=False)
                        break
                    
                    # Directly append to llm_output, reducing queue operations
                    llm_output += str(output)
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

                # llm_message.update_content(content=llm_output_total)
                self.database.add_data(user_input, "user")
                self.database.add_data(llm_output_total, "bot")
                llm_output_total = ""

    def agent_memory_output_ws(
            self, 
            agent = None, 
            is_llm_ready_event = None, 
            prompt_template = None, 
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

                memory = self.database.search_data(query=[user_input])
            
                self.speaking_event.set()
                user_input = "User question: " + user_input + ", Memory: " + memory
                print(f"\033[95mUser: {user_input} \033[0m")  # 紫色高亮输出
                # agent.invoke(prompt_template.format(user_input=user_input, memory=memory))
                agent.invoke(user_input)
        except KeyboardInterrupt:
            print("agent_output_ws KeyboardInterrupt\n")
            self.stop_event.set()
            
            # torch.cuda.ipc_collect()
            print("User stopped the program\n")

    def llm_memory_output_ws(
            self, 
            model = None, 
            is_llm_ready_event = None, 
            prompt_template = None, 
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

                memory = self.database.search_data(query=[user_input])
                
                self.speaking_event.set()
                llm_output = ""
                llm_output_total = ""
                is_llm_thinking = False

                user_input = "User question: " + user_input + ", Memory: " + memory
                print(f"\033[95mUser: {user_input} \033[0m")  # 紫色高亮输出
                # for output in model.stream(prompt_template.format(user_input=user_input, memory=memory)):
                for output in model.stream(user_input):
                    if self.is_user_talking.is_set() or not self.user_input_queue.empty():
                        if not self.llm_output_queue.empty():
                            empty_queue = self.llm_output_queue.get(block=False)
                        break
                    
                    # Directly append to llm_output, reducing queue operations
                    llm_output += str(output)
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

                # llm_message.update_content(content=llm_output_total)
                self.database.add_data(user_input, "user")
                self.database.add_data(llm_output_total, "bot")
                llm_output_total = ""

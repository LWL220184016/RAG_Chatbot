import queue
import time
import json

from Data_Storage.database import Database_Handler
from Data_Storage.json_memory import JSON_Memory
from tenacity import retry, stop_after_attempt, wait_fixed
from LLM.prompt_template import Message

class LLM:
    def __init__(
            self, 
            model_name: str = "deepseek-r1:14b", 
            is_user_talking = None, 
            stop_event = None, 
            speaking_event = None, 
            user_input_queue: queue = None, 
            llm_output_queue: queue = None, 
            llm_output_queue_ws: queue = None, 
            tools = [], 
            database: Database_Handler = None,
            use_temp_memory: bool = True, 
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
        self.chat_history_recorder = JSON_Memory("chat_history_record")
        
        # Initialize Redis memory handler if config is provided
        if use_temp_memory:
            self.temp_memory_handler = JSON_Memory("temp_memory")
    
    def langchain_agent_output_ws( 
            self, 
            agent = None, 
            is_llm_ready_event = None, 
            session_id = "default"
        ): 

        print("llm waiting text")
        is_llm_ready_event.set()
        user_last_talk_time = time.time()
        user_msg = Message(user_role="user")

        try:
            while not self.stop_event.is_set():
                user_input, updated_user_msg = self.get_user_input(user_msg, user_last_talk_time)
                llm_output = agent.invoke(updated_user_msg)
                if self.temp_memory_handler:
                    self.temp_memory_handler.add(message=user_input, Role="user")
                    self.temp_memory_handler.add(message=llm_output.get("output"), Role="assistant")
                
                # Store LLM output in temporary memory if Redis is configured
                self.chat_history_recorder.add_no_limit(message=user_input, Role="user")
                self.chat_history_recorder.add_no_limit(message=llm_output.get("output"), Role="assistant")
                if self.database is not None:
                    self.database.add_data(user_input, "user")
                    self.database.add_data(llm_output.get("output"), "llm")
        except KeyboardInterrupt:
            print("langchain_agent_output_ws KeyboardInterrupt\n")
            self.stop_event.set()
            
            # torch.cuda.ipc_collect()
            print("User stopped the program\n")

    def llm_output_ws(
            self, 
            model = None, 
            is_llm_ready_event = None, 
            session_id = "default" 
        ): 
        """
        LLM output processing

        llm_output: 目的在於把模型輸出的内容根據逗號或者句號進行分割，儘早讓 TTS 處理以節省時間
        llm_output_total: 目的在於儲存模型這次對話輸出的所有内容，用於記憶儲存以及輸出顯示

        目前 LLM 的流式輸出存在問題，實際上是一次性輸出所有内容，也并沒有根據逗號和句號進行分割，需要以後實現

        """

        print("llm waiting text")
        is_llm_ready_event.set()
        user_last_talk_time = time.time()
        user_msg = Message(user_role="user")

        while not self.stop_event.is_set():
            user_input, updated_user_msg = self.get_user_input(user_msg, user_last_talk_time)
            llm_output_total = ""
            for thinking_output, output in model.stream(updated_user_msg):
                
                self.speaking_event.set()
                llm_output = ""
                is_llm_thinking = False

                if self.is_user_talking.is_set() or not self.user_input_queue.empty():
                    if not self.llm_output_queue.empty():
                        # Empty the queue
                        empty_queue = self.llm_output_queue.get(block=False)
                    break
                
                # Directly append to llm_output_total, reducing queue operations
                llm_output_total += str(output)

            self.llm_output_queue.put(llm_output_total)
            self.llm_output_queue_ws.put(llm_output_total)
            # Store LLM output in temporary memory if Redis is configured
            # self.chat_history_recorder.add_no_limit(user_message=user_input, llm_message=llm_output_total.get("output"))
            self.chat_history_recorder.add_no_limit(message=user_input, Role="user")
            self.chat_history_recorder.add_no_limit(message=llm_output_total, Role="assistant")
            if self.temp_memory_handler and llm_output_total:
                self.temp_memory_handler.add(message=user_input, Role="user")
                self.temp_memory_handler.add(message=llm_output_total, Role="assistant")
                
            # llm_message.update_content(content=llm_output_total)
            if self.database is not None:
                self.database.add_data(user_input, "user")
                self.database.add_data(llm_output_total, "bot")
            llm_output_total = ""

    def clear_temp_memory(self, session_id="default"):
        """Clear temporary memory for a specific session"""
        if self.temp_memory_handler:
            self.temp_memory_handler.clear_conversation(session_id)
            return True
        return False

    def get_user_input(self, user_msg: Message, user_last_talk_time: float):
        """Get user input from the queue"""
        try:
            # user_input = self.user_input_queue.get(timeout=0.1)
            user_input = self.user_input_queue.get()
        except queue.Empty:
            user_input = " "

        if not user_input == " ":
            print(f"\033[95mUser: {user_input} \033[0m")  # 紫色高亮输出

        # Prepare streaming input with context if Redis is configured
        if self.temp_memory_handler:
            recent_history = self.temp_memory_handler.get()
            print(f"\033[95mrecent_history: {recent_history} \033[0m")  # 紫色高亮输出
            return user_input, user_msg.update_content(content=user_input, memory=recent_history)
        
        else:
            return user_input, user_msg.update_content(content=user_input)
    
    def get_user_input_stream(self, user_msg: Message, user_last_talk_time: float):
        """Get user input from the queue"""
        user_input = ""
        while user_input == "":
            if not self.is_user_talking.is_set():
                if time.time() - user_last_talk_time > 5:
                    user_input = ""
                try:
                    user_input += self.user_input_queue.get(timeout=0.1) + " "
                    time.sleep(0.1)  # Avoid busy waiting
                except queue.Empty:
                        time.sleep(0.1)  # Avoid busy waiting
                        continue
                if not self.user_input_queue.empty():
                    user_input += self.user_input_queue.get() + " "
                    continue
            else: # user is talking
                user_last_talk_time = time.time()
                continue  # Skip if the user is talking

        print(f"\033[95mUser: {user_input} \033[0m")  # 紫色高亮输出


        # Prepare streaming input with context if Redis is configured
        if self.temp_memory_handler:
            recent_history = self.temp_memory_handler.get()
            print(f"\033[95mrecent_history: {recent_history} \033[0m")  # 紫色高亮输出
            return user_msg.update_content(content=user_input, memory=recent_history)
        
        else:
            return user_msg.update_content(content=user_input)

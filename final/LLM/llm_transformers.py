import torch
import threading
import queue

from LLM.llm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain.agents import AgentType, initialize_agent


class LLM_Transformers(LLM):
    def __init__(
        self, 
        torch_dtype = torch.float16, 
        device: str = "cuda:0", 
        trust_remote_code: bool = False, 
        load_in_8bit: bool = False, 
        load_in_4bit: bool = True, 

        # model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
        # model_name: str = "google/gemma-3-12b-it", 
        # model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        model_name: str = "Qwen/Qwen3-1.7B", 
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
        self.device = device

        if load_in_8bit and load_in_4bit:
            raise ValueError("Can only choose one (8bit or 4bit)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 显式设置 pad_token_id (Explicitly set pad_token_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        self.streamer = TextStreamer(self.tokenizer)

        self.model.stream = self.stream

    def stream(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device) # ne: not equal

        output_tokens = [] # 缓存生成的 tokens (Cache generated tokens)
        for output in self.model.generate(input_ids, attention_mask=attention_mask, streamer=self.streamer, pad_token_id=self.tokenizer.pad_token_id): # 传入 attention_mask 和 pad_token_id (Pass attention_mask and pad_token_id)
            output_tokens.extend(output.tolist())
            current_output_text = self.tokenizer.decode(output, skip_special_tokens=True) # 解码当前输出的 tokens (Decode current output tokens)
            yield current_output_text

    def langchain_agent_output_ws(
            self,
            is_llm_ready_event, 
        ):
        
        raise NotImplementedError("langchain_agent_output_ws() in llm_transformers is not implemented yet.")

    def llm_output_ws(
            self,
            is_llm_ready_event: threading.Event,
        ):
        
        # 在這裡傳遞必要的參數給父類別的方法
        super().llm_output_ws(self.model, is_llm_ready_event)

        # # TODO
        # problem in deepseek-ai/DeepSeek-R1-Distill-Qwen-7B llm output and 
        # faster whisper streaming words not extend and too slow

# TODO
#         print("llm waiting text")
#         is_llm_ready_event.set()
#         user_input = ""
#         user_last_talk_time = time.time()

#         while not self.stop_event.is_set():
#             if not self.is_user_talking.is_set():
#                 if time.time() - user_last_talk_time > 5:
#                     user_input = ""
#                 try:
#                     user_input += self.user_input_queue.get(timeout=0.1) + " "
#                 except queue.Empty:
#                     continue
#                 if not self.user_input_queue.empty():
#                     user_input += self.user_input_queue.get() + " "
#             else: # user is talking
#                 user_last_talk_time = time.time()
#                 continue  # Skip if the user is talking

#             print("user_input: " + user_input + "  -----------------------------------------------------user_input")

#             prompt = "return the previous chat content relate to the queue"
#             # memory = rag.search_rag(query=user_input, prompt=prompt, mode="hybrid")

#             # Assuming 'update_content' method exists for Message class
#             # msg = user_message.update_content(content=user_input, memory=memory)

#             self.speaking_event.set()
#             llm_output = ""
#             llm_output_total = ""
#             is_llm_thinking = False

#             output_tokens = [] # 缓存生成的 tokens (Cache generated tokens)
#             input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
#             #  创建 attention_mask (Create attention_mask)
#             attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device) # ne: not equal

#             for output in self.model.generate(input_ids, attention_mask=attention_mask, streamer=self.streamer, pad_token_id=self.tokenizer.pad_token_id): # 传入 attention_mask 和 pad_token_id (Pass attention_mask and pad_token_id)
#                 if self.is_user_talking.is_set() or not self.user_input_queue.empty():
#                     if not self.llm_output_queue.empty():
#                         empty_queue = self.llm_output_queue.get(block=False)
#                     break
#                 #  output 是 token ids,  extend 到 output_tokens 列表中 (output is token ids, extend to output_tokens list)
#                 output_tokens.extend(output.tolist())
#                 current_output_text = self.tokenizer.decode(output, skip_special_tokens=True) # 解码当前输出的 tokens (Decode current output tokens)


#                 llm_output += current_output_text
#                 if current_output_text == "<think>" and not is_llm_thinking:
#                     is_llm_thinking = True
#                     print("is_llm_thinking = True")
#                 elif current_output_text == "</think>" and is_llm_thinking:
#                     is_llm_thinking = False
#                     print("is_llm_thinking = False")
#                 if current_output_text in ["，", ",", "。", ".", "？", "?", "！", "!"] or "</think>" in current_output_text:
#                     llm_output_total += llm_output
#                     print("llm output: " + llm_output)
#                     if not is_llm_thinking or "</think>" in current_output_text:
#                         self.llm_output_queue.put(llm_output)
#                         print("after put llm_output_queue: ", self.llm_output_queue.qsize())
#                     self.llm_output_queue_ws.put(llm_output)
#                     llm_output = ""

#             #  在循环结束后，解码完整的 token 序列 (Decode the complete token sequence after the loop)
#             #  final_output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
#             #  print("Final LLM Output:", final_output_text)


#             # Assuming 'update_content' method exists for Message class
#             # llm_message.update_content(content=llm_output_total)
#             llm_output_total = ""


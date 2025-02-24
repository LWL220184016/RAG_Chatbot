from langchain.callbacks.base import BaseCallbackHandler
from collections import deque

class OllamaAgentStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, is_user_talking, user_input_queue, llm_output_queue, llm_output_queue_ws):
        self.is_user_talking = is_user_talking
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        self.llm_output = ""  # 用于缓存分段响应然後輸入 tts
        self.full_response = ""  # 用于缓存完整响应
        # self.is_final_answer = False
        self.is_put_to_llm_output_queue = False
        self.token_window = deque(maxlen=9)  # 滑动窗口
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # # 流式输出每个 Token（Ollama 的 token 可能包含格式字符）
        # self.full_response += token
        # print(f"\033[95m|\033[0m", end="", flush=True)  # 紫色高亮输出

        # # 更新滑动窗口
        # self.token_window.append(token)
        # self.llm_output_queue_ws.put(token)
        # # 检查滑动窗口中的 token 是否匹配 ' "Final Answer",\n'
        # if not self.is_final_answer and "".join(self.token_window) == 'Final Answer",\n  "action_input": "':
        #     self.is_final_answer = True
        #     self.is_put_to_llm_output_queue = True
        #     print("\n\033[91m🤖 Action: Final Answer\033[0m") #

        # elif self.is_final_answer:
        #     if self.is_user_talking.is_set() or not self.user_input_queue.empty():
        #         if not self.llm_output_queue.empty():
        #             empty_queue = self.llm_output_queue.get(block=False)
        #         return
            
        #     # Directly append to llm_output, reducing queue operations
        #     self.llm_output += token

        #     if '"' in token:
        #         self.is_final_answer = False
        #         return

        #     if any(punct in token for punct in ["，", ",", "。", ".", "？", "?", "！", "!"]):
        #         self.llm_output_queue.put(self.llm_output)
        #         self.llm_output = ""
            # self.neo4j.add_dialogue_record(user_message, llm_message)
        pass

# 2. 嘗試不用 langchain 的情況下通過提示詞嘗試讓模型生成 json 或者 code 的 tools 呼叫
# 3. 在抱抱臉的 dc 群組中詢問我對 langchain 的理解是不是正確的，現在 langchain 表現不好是否因爲我對 langchain 的使用錯誤
# 4. 通過修改提示詞模板和 OllamaAgentStreamingCallbackHandler 來解決 ollama 運行的模型有時候輸出 </think> 有時候沒有輸出的問題，這會導致無法吧正確的内容輸出到 tts
5. 搜索的信息中嘗試包含食物或者食譜的圖片
6. 在食譜和食物的數據頁面添加一個按鈕，當用戶點擊時，會生成總結
7. MemGPT for llm memory


    def on_agent_action(self, action, **kwargs):
        # Agent 调用工具时触发
        # print(f"\n\033[94m🤖 Action: {action.log}\033[0m")  # 蓝色高亮
        # print(f"\n\033[91m🤖 Action: {action.log}\033[0m")  # 红色高亮
        # self.queue.put(f"\nAction: {action.log}")
        pass

    def on_tool_end(self, output: str, **kwargs):
        # 工具执行完成
        # print(f"\n\033[93m🔍 Observation: {output}\033[0m")  # 黄色高亮
        # print("on_tool_end called")
        # print(f"\n\033[38;5;208m🔍 Observation: {output}\033[0m")  # 橙色高亮 (256-color)
        # self.queue.put(f"\nObservation: {output}")
        pass

    def on_agent_finish(self, finish, **kwargs):
        # Agent 完成所有操作
        output = finish.return_values['output']
        print(f"\n\033[38;5;208m🔍 return_values['output']: {output}\033[0m")  # 橙色高亮 (256-color)
        llm_output = ""

        for words in output:
            if self.is_user_talking.is_set() or not self.user_input_queue.empty():
                if not self.llm_output_queue.empty():
                    empty_queue = self.llm_output_queue.get(block=False)
                break
            
            # Directly append to llm_output, reducing queue operations
            llm_output += words

            if words in ["，", ",", "。", ".", "？", "?", "！", "!"] or "</think>" in words:
                
                if self.is_put_to_llm_output_queue:
                    self.llm_output_queue.put(llm_output)
                self.llm_output_queue_ws.put(llm_output)
                if "</think>" in llm_output: self.is_put_to_llm_output_queue = True
                if "<|IS|>" in llm_output: self.is_put_to_llm_output_queue = False

                # print("llm words: " + llm_output, "  self.llm_output_queue: " + str(self.llm_output_queue.qsize()))
                llm_output = ""

        # self.neo4j.add_dialogue_record(user_message, llm_message)
        pass

    def on_error(self, error, **kwargs):
        print(f"\n🔥 Error: {str(error)}")

class GoogleAgentStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, is_user_talking, user_input_queue, llm_output_queue, llm_output_queue_ws):
        self.is_user_talking = is_user_talking
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws

    def on_agent_action(self, action, **kwargs):
        # Agent 调用工具时触发
        # print(f"\n\033[94m🤖 Action: {action.log}\033[0m")  # 蓝色高亮
        # print(f"\n\033[91m🤖 Action: {action.log}\033[0m")  # 红色高亮
        # self.llm_output_queue.put(f"\nAction: {action.log}")
        pass

    def on_agent_finish(self, finish, **kwargs):
        # Agent 完成所有操作
        output = finish.return_values['output']
        print(f"\n\033[38;5;208m🔍 return_values['output']: {output}\033[0m")  # 橙色高亮 (256-color)
        self.llm_output_queue_ws.put(output)
        llm_output = ""

        for words in output:
            if self.is_user_talking.is_set() or not self.user_input_queue.empty():
                if not self.llm_output_queue.empty():
                    empty_queue = self.llm_output_queue.get(block=False)
                break
            
            # Directly append to llm_output, reducing queue operations
            llm_output += words
            if "<|IS|>" in llm_output: break

            if words in ["，", ",", "。", ".", "？", "?", "！", "!"]:
                self.llm_output_queue.put(llm_output)
                # print("llm words: " + llm_output, "  self.llm_output_queue: " + str(self.llm_output_queue.qsize()))
                llm_output = ""

        # self.neo4j.add_dialogue_record(user_message, llm_message)
        pass
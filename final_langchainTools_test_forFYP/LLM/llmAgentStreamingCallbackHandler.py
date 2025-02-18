from langchain.callbacks.base import BaseCallbackHandler

class OllamaAgentStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, is_user_talking, user_input_queue, llm_output_queue, llm_output_queue_ws):
        self.is_user_talking = is_user_talking
        self.user_input_queue = user_input_queue
        self.llm_output_queue = llm_output_queue
        self.llm_output_queue_ws = llm_output_queue_ws
        self.llm_output = ""  # 用于缓存分段响应然後輸入 tts
        self.full_response = ""  # 用于缓存完整响应
        self.is_agent_action = False
        self.is_llm_thinking = False
        self.token_window = []  # 滑动窗口
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 流式输出每个 Token（Ollama 的 token 可能包含格式字符）
        self.full_response += token
        # print(f"\033[95m{token}\033[0m", end="", flush=True)  # 紫色高亮输出

        # 更新滑动窗口
        self.token_window.append(token)
        if len(self.token_window) > 4:  # 假设 ' "Final Answer",\n' 是由 4 个 token 组成
            self.token_window.pop(0)

        # 检查滑动窗口中的 token 是否匹配 ' "Final Answer",\n'
        print("----------------" + "".join(self.token_window) + "----------------")
        if "".join(self.token_window) == ' "Final Answer",\n':
            self.is_agent_action = True
            print("\n\033[91m🤖 Action: Final Answer\033[0m") #

        if self.is_agent_action:
            if self.is_user_talking.is_set() or not self.user_input_queue.empty():
                if not self.llm_output_queue.empty():
                    empty_queue = self.llm_output_queue.get(block=False)
                return
            
            # Directly append to llm_output, reducing queue operations
            self.llm_output += token
            if "<think>" in token and not self.is_llm_thinking:
                self.is_llm_thinking = True
                print("self.is_llm_thinking = True")

            elif "</think>" in token and self.is_llm_thinking:
                self.is_llm_thinking = False
                print("self.is_llm_thinking = False")

            if token in ["，", ",", "。", ".", "？", "?", "！", "!"] or "</think>" in token:
                print("\n\n   ---llm token: " + self.llm_output + "---\n\n")
                if not self.is_llm_thinking and "</think>" in token:
                    self.llm_output_queue.put(self.llm_output)
                self.llm_output_queue_ws.put(self.llm_output)
                llm_output = ""

            # self.neo4j.add_dialogue_record(user_message, llm_message)

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
        # print(f"\n\033[95m✅ Final Answer: {finish.return_values['output']}\033[0m")
        # self.queue.put(f"\nFinal Result: {finish.return_values['output']}")
        # self.queue.put(None)  # 结束信号
        self.is_agent_action = False
        pass

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
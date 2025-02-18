from langchain.callbacks.base import BaseCallbackHandler

class LLMAgentStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.full_response = ""  # 用于缓存完整响应
        self.is_agent_action = False
        self.token_window = []  # 滑动窗口
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 流式输出每个 Token（Ollama 的 token 可能包含格式字符）
        self.full_response += token
        print(f"\033[95m{token}\033[0m", end="", flush=True)  # 紫色高亮输出

        # 更新滑动窗口
        self.token_window.append(token)
        if len(self.token_window) > 4:  # 假设 "Final Answer" 是由 3 个 token 组成
            self.token_window.pop(0)

        # 检查滑动窗口中的 token 是否匹配 "Final Answer"
        print("----------------" + "".join(self.token_window) + "----------------")
        if "".join(self.token_window) == ' "Final Answer",\n':
            self.is_agent_action = True
            print("\n\033[91m🤖 Action: Final Answer\033[0m") #

            self.queue.put(token)

# ```json
# {
#   "action": "Final Answer",
#   "action_input": "最新的美国总统是乔·拜登（Joe Biden），他自2021年1月20日起担任这一职位。"
# }
# ```

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

# class QueueCallbackHandler(BaseCallbackHandler):
#     """将回调数据存入队列供生成器读取"""
#     def __init__(self, queue):
#         self.queue = queue

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.queue.put(token)

#     def on_agent_action(self, action, **kwargs):
#         self.queue.put(f"\nAction: {action.log}")

#     def on_tool_end(self, output: str, **kwargs):
#         self.queue.put(f"\nObservation: {output}")

#     def on_agent_finish(self, finish, **kwargs):
#         self.queue.put(f"\nFinal Result: {finish.return_values['output']}")
#         self.queue.put(None)  # 结束信号

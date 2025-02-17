from langchain.callbacks.base import BaseCallbackHandler

class LLMAgentStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.full_response = ""  # 用于缓存完整响应
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 流式输出每个 Token（Ollama 的 token 可能包含格式字符）
        self.full_response += token
        # print(f"\033[92m{token}\033[0m", end="", flush=True)  # 绿色高亮输出
        print(f"\033[95m{token}\033[0m", end="", flush=True)  # 紫色高亮输出
        self.queue.put(token)
        print("<new token> queue size: ", self.queue.qsize(), "=== token: ", token)

    def on_agent_action(self, action, **kwargs):
        # Agent 调用工具时触发
        # print(f"\n\033[94m🤖 Action: {action.log}\033[0m")  # 蓝色高亮
        print(f"\n\033[91m🤖 Action: {action.log}\033[0m")  # 红色高亮
        self.queue.put(f"\nAction: {action.log}")
        print("<new action>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def on_tool_end(self, output: str, **kwargs):
        # 工具执行完成
        # print(f"\n\033[93m🔍 Observation: {output}\033[0m")  # 黄色高亮
        print("on_tool_end called")
        print(f"\n\033[38;5;208m🔍 Observation: {output}\033[0m")  # 橙色高亮 (256-color)

        self.queue.put(f"\nObservation: {output}")

    def on_agent_finish(self, finish, **kwargs):
        # Agent 完成所有操作
        print(f"\n\033[95m✅ Final Answer: {finish.return_values['output']}\033[0m")
        self.queue.put(f"\nFinal Result: {finish.return_values['output']}")
        self.queue.put(None)  # 结束信号

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

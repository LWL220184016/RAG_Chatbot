from langchain.callbacks.base import BaseCallbackHandler

class LLMAgentStreamingCallbackHandler(BaseCallbackHandler):
    def ___init__(self):
        self.full_response = ""  # 用于缓存完整响应
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 流式输出每个 Token（Ollama 的 token 可能包含格式字符）
        self.full_response += token
        print(f"\033[92m{token}\033[0m", end="", flush=True)  # 绿色高亮输出

    def on_agent_action(self, action, **kwargs):
        # Agent 调用工具时触发
        print(f"\n\033[94m🤖 Action: {action.log}\033[0m")  # 蓝色高亮

    def on_tool_end(self, output: str, **kwargs):
        # 工具执行完成
        print(f"\n\033[93m🔍 Observation: {output}\033[0m")  # 黄色高亮

    def on_agent_finish(self, finish, **kwargs):
        # Agent 完成所有操作
        print(f"\n\033[92m✅ Final Answer: {finish.return_values['output']}\033[0m")
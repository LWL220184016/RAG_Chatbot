from memgpt import MemGPT
from langchain.memory import BaseMemory
from typing import List, Dict

class MemGPTMemory(BaseMemory):
    def __init__(self, memory_id="langchain_memory"):
        # 初始化 MemGPT 客戶端
        self.client = MemGPT()
        # 創建一個純記憶用的 agent，避免與 LangChain 的 agent 混淆
        self.memory_store = self.client.create_agent(
            name=memory_id,
            # 可選：指定不使用 LLM，只作為記憶存儲
            # 如果 MemGPT API 支援純記憶模式，可以進一步簡化
        )

    def load_memory_variables(self, inputs: Dict[str, any]) -> Dict[str, any]:
        # 從 MemGPT 獲取記憶內容
        history = self.client.get_conversation(self.memory_store)
        return {"history": history}

    def save_context(self, inputs: Dict[str, any], outputs: Dict[str, any]) -> None:
        # 將對話上下文保存到 MemGPT
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")
        self.client.send_message(self.memory_store, input_text, role="user")
        self.client.send_message(self.memory_store, output_text, role="assistant")

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

# 示例：與 LangChain agent 整合
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = [Tool(name="Search", func=lambda x: "Search result", description="A search tool")]

# 使用自定義記憶
memory = MemGPTMemory(memory_id="my_unique_memory")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

# 測試
print(agent.run("Hi, tell me something interesting."))
print(agent.run("What did I just ask you?"))
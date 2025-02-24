import letta
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory.chat_memory import BaseChatMemory  # 修改 import 行
from langchain.prompts import PromptTemplate

# 自訂一個基於 LettA 的記憶管理器，繼承自 LangChain 的 BaseChatMemory
class LettaMemory(BaseChatMemory): # 修改繼承的類別
    """
    此類別利用 LettA 庫來管理 agent 的記憶。
    你需要根據 LettA 提供的 API 實作 get_history 與 add_history 方法。
    """

    def __init__(self):
        # 初始化 LettA 的記憶庫物件（依據 LettA 的實際 API 來初始化）
        # 這個示例假設 LettA 提供 MemoryStore 類別
        self.let_memory = letta.MemoryStore()

    def load_memory_variables(self, inputs: dict) -> dict:
        """
        返回目前的對話歷史，作為 prompt 的上下文變數（例如 "history"）。
        """
        history = self.let_memory.get_history()  # 假設 get_history() 返回已儲存的對話字串
        return {"history": history}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """
        將本次對話的輸入與輸出存入 LettA 的記憶庫中。
        """
        # 根據需求定義儲存的格式，這裡將使用者輸入與 AI 回答串接後儲存
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        # 假設 LettA 的記憶庫有 add_history 方法
        self.let_memory.add_history(f"使用者：{user_input}\nAssistant：{ai_output}\n")

    @property
    def memory_keys(self):
        """指定此記憶管理器所處理的變數名稱，本例中為 'history'"""
        return ["history"]

# 示例：建立一個基於 LangChain 的 agent，並使用 LettaMemory 管理記憶
def main():
    # 初始化 LLM，請根據實際需要設定 API 金鑰或環境變數

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=10,
        max_retries=6,
        streaming=True,  # 启用流式传输
        callbacks=[],  # 标准输出回调
    )
    # 初始化自訂記憶管理器
    memory = LettaMemory()

    # 可以定義自訂 prompt，這裡我們假設把對話歷史作爲輸入上下文
    template = (
        "下麵是之前對話的歷史：\n{history}\n"
        "你現在收到使用者的新問題：\n{input}\n"
        "請提供回答："
    )
    prompt = PromptTemplate(template=template, input_variables=["history", "input"])

    # 初始化 agent，這裡使用 Zero-Shot ReAct Prompt 方式（可根據需求調整 AgentType）
    agent = initialize_agent(
        tools=[],  # 如有工具可傳入；此處為空列表
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,  # 將 LettA 記憶管理器傳入
        prompt=prompt
    )

    # 執行一次對話，agent 會自動調用 memory 的 load/save 方法來取得與維護對話上下文
    # question = "請問 6+7 等於多少？"
    # print("使用者問題：", question)
    # response = agent.run(question)
    # print("Assistant 回答：", response)

# 測試
    print(agent.run("Hi, tell me something interesting."))
    print(agent.run("What did I just ask you?"))

if __name__ == "__main__":
    main()
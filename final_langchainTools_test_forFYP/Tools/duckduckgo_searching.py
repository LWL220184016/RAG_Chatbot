# 安装必要库（先执行）
# pip install duckduckgo-search langchain

from langchain.tools import tool
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Optional, List, Dict

class DuckDuckGoSearchWrapper:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.session = DDGS()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True
    )
    def search_with_retry(self, query: str) -> List[Dict]:
        """执行带有指数退避重试的搜索"""
        try:
            results = []
            # 使用DDGS的文本搜索功能
            for result in self.session.text(query):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "description": result.get("body", "")
                })
                if len(results) >= self.max_results:
                    break
            return results
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo搜索失败: {str(e)}")

# 示例使用
if __name__ == "__main__":
    from tool import Tools
    tool = Tools()
    # 测试搜索
    print(tool.duckduckgo_search.run("最新AI发展 news", 2))
    
    # 整合到LangChain Agent
    from langchain.agents import initialize_agent
    from langchain_ollama import OllamaLLM
    
    llm = OllamaLLM(
        # model="llama3-groq-tool-use",
        model="deepseek-r1_14b_FYP4",
        # model="llama3.2:1b",
        # model="deepseek-r1:32b",
        temperature=0.1,
        top_p=0.9,
    )   
    
    agent = initialize_agent(
        tools=[tool.duckduckgo_search],
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True
    )
    
    response = agent.run("OpenAI最近发布了什么重要更新？")
    print(response)
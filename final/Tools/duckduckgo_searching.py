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

@tool
def duckduckgo_search(
    query: str,
    max_results: Optional[int] = 3
) -> str:
    """
    使用DuckDuckGo执行网络搜索，适用于需要实时信息的查询。
    
    参数：
    - query: 搜索关键词（必须用英文逗号分隔的关键词组合）
    - max_results: 返回的最大结果数量（默认3，最大5）
    
    返回格式：
    [结果1标题](链接1)
    摘要: 结果1摘要...
    
    [结果2标题](链接2)
    摘要: 结果2摘要...
    """
    try:
        # 参数验证
        if not isinstance(query, str) or len(query.strip()) == 0:
            return "错误：搜索关键词不能为空"
        
        max_results = min(int(max_results), 5) if max_results else 3
        
        # 执行搜索
        search = DuckDuckGoSearchWrapper(max_results=max_results)
        results = search.search_with_retry(query)
        
        # 格式化结果
        formatted = []
        for idx, result in enumerate(results, 1):
            formatted.append(
                f"[{result['title']}]({result['url']})\n"
                f"摘要: {result['description'][:150]}..."
            )
            
        return "\n\n".join(formatted) if formatted else "未找到相关结果"
        
    except Exception as e:
        return f"搜索失败：{str(e)}"

# 示例使用
if __name__ == "__main__":
    # 测试搜索
    print(duckduckgo_search.run("最新AI发展 news", 2))
    
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
        tools=[duckduckgo_search],
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True
    )
    
    response = agent.run("OpenAI最近发布了什么重要更新？")
    print(response)
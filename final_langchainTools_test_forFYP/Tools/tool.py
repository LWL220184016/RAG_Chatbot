import sys
sys.path.append('final_langchainTools_test_forFYP/Data_Storage')

from Data_Storage.qdrant import Qdrant_Handler
from Tools.duckduckgo_searching import DuckDuckGoSearchWrapper
from typing import Optional, List, Dict
from langchain.tools import tool, Tool
from tenacity import retry, stop_after_attempt, wait_random_exponential


# class tools(Tool):
class Tools:
    def __init__(
            self, 
            database_qdrant: Qdrant_Handler = None, 
        ):
        
        self.database_qdrant = database_qdrant

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


    @tool
    def querying_qdrant(
        self, 
        query: str,
        user_filter: dict = None, 
        collection_name: str = None
    ) -> str:
        """
        在向量資料庫 Qdrant 中搜尋數據，适用于需要歷史對話信息的查询。
        
        参数：
        - query: 搜索关键词（目前每次搜尋僅允許一個關鍵詞)
        - user_filter: 搜索过滤器，支援兩個屬性:
            1. "timestamp" 範例值: "2022-01-01T00:00:00"
            2. "speaker" 範例值: "user"
        - collection_name: 搜索的集合名称（chat_YYYY_MM）

        返回：
        包含搜索结果的JSON字符串，包含以下數據:
            1. "message": "消息内容",
            2. "timestamp": "时间戳",
            3. "speaker": "发言者",
            4. "similarity_score": 相似度分数
        """

        return self.database_qdrant.search_data(
                query, 
                user_filter, 
                collection_name, 
            ) 
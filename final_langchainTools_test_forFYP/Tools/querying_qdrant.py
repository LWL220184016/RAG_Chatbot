import sys
sys.path.append('final_langchainTools_test_forFYP/Data_Storage')

from Data_Storage.qdrant import Qdrant_Handler
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_random_exponential

@tool
def querying_qdrant(
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

    return Qdrant_Handler.search_data(
               query, 
               user_filter, 
               collection_name
           )
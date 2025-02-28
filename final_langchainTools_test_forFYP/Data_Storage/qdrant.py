import os
import datetime
import json

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from Data_Storage.embedding_model.embedder import Embedder
from Data_Storage.database import Database_Handler

class Qdrant_Handler(Database_Handler):
    def __init__(
            self, 
            vector_size: int = 768, 
            embedder: Embedder = Embedder(), 
            dataID: int = 0,
        ):

        super().__init__()
        self.client = QdrantClient(
            host = os.getenv('QDRANT_HOST'), 
            port = os.getenv('QDRANT_PORT'), 
        )

        self.vector_size = vector_size  # For demonstration, we use a low-dimensional vector
        self.embedder = embedder
        self.dataID = dataID
        self.create_dataset()

    def create_dataset(
            self, 
            collection_name: str = None, 
        ):

        if collection_name is None:
            collection_name = Database_Handler.get_newest_chat_name()

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": self.vector_size,
                    "distance": Distance.COSINE
                }
            )
            print(f"Collection {collection_name} created successfully.")
        else:
            print(f"Collection {collection_name} already exists.")

    def add_data(
            self, 
            msg: str, 
            user_role: str, 
            collection_name: str = None, 
        ):
        
        if collection_name is None:
            collection_name = Database_Handler.get_newest_chat_name()

        self.dataID += 1
        embeddings = self.embedder.embed([msg])
        # print(f"Vector: {embeddings}")
        time = datetime.datetime.now().isoformat()
        meta_data = {
            "timestamp": time, 
            "speaker": user_role, 
            # "metadata": { 
            #     "length": len(msg), 
            # } 
        }
        points = {
            "id": self.dataID,
            "vector": embeddings[0].tolist(),
            "payload": {
                "msg": msg,
                **meta_data  # Merge additional JSON data into the payload
            }
        }

        # Insert (upsert) the points into the collection
        self.client.upsert(collection_name=collection_name, points=[points])
        
        print(f"""
            ====================================================================
            Action: Qdrant_Handler.add_data 
            dataID: {self.dataID}
            Msg: {msg}
            Vector size: {embeddings.shape}
            Status: Data inserted successfully.
            ====================================================================
        """)

    def search_data(
            self, 
            query: str, 
            user_filter: dict = None, 
            collection_name: str = None, 
            max_results: int = 2, 
        ): 
        """
        在向量資料庫 Qdrant 中搜尋數據，适用于需要歷史對話信息的查询。
        
        参数:
        - query: 搜索关键词（目前每次搜尋僅允許一個關鍵詞, 如有需要請進行多次搜索）
        - user_filter: 搜索过滤器（默认无）
            如要使用, 請在呼叫函數的參數中包含以下格式數據:
            user_filter = {
                "timestamp": "2022-01-01T00:00:00", 
                "speaker": "user"
            }
        - collection_name: 搜索的集合名称（默认最新集合 chat_YYYY_MM）
            例如: collection_name = "chat_2022_01"
        - max_results: 返回的最大结果数量（默认2）

        返回:
        包含搜索结果的JSON字符串，格式如下:
        [
            {
                "message": "消息内容",
                "timestamp": "时间戳",
                "speaker": "发言者",
                "similarity_score": 相似度分数
            },
            {...}
        ]
        """

        if collection_name is None:
            collection_name = Database_Handler.get_newest_chat_name()

        query_vector = self.embedder.embed(query)
        filter_conditions = []

        if user_filter:
            for key, value in user_filter.items():
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        filter_condition = Filter(must=filter_conditions) if filter_conditions else None

        # Perform a search for the top [max_results] nearest neighbors
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector[0],
            limit=max_results,
            query_filter=filter_condition
        )

        # Print out the search results
        print("Search results:")
        memory = []

        for point in search_result:
            if isinstance(point, tuple):
                # If the result is a tuple, access elements by index
                point = point[1]
                for p in point:
                    memory.append({
                        "message": p.payload['msg'],
                        "timestamp": p.payload['timestamp'],
                        "speaker": p.payload['speaker'],
                        "similarity_score": p.score,
                    })
        memory = sorted(memory, key=lambda x: x['similarity_score'], reverse=True)
        memory = json.dumps(memory, indent=4)
        print(memory)
        return memory
import os
import datetime
import json

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from typing import Optional, Dict
try:
    from Data_Storage.embedding_model.embedder import Embedder
    from Data_Storage.database import Database_Handler
except Exception:
    from embedding_model.embedder import Embedder
    from database import Database_Handler


# export QDRANT_HOST=localhost
# export QDRANT_PORT=6333

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
            Speaker: {user_role}
            Vector size: {embeddings.shape}
            Status: Data inserted successfully.
            ====================================================================
        """)

    def search_data(
            self, 
            query: str, 
            user_filter: Optional[dict] = None, 
            collection_name: Optional[str] = None, 
            max_results: int = 5, 
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

        if user_filter is not None:
            for key, value in user_filter.items():
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchText(text=value)
                    )
                )
        filter_condition = Filter(must=filter_conditions) if filter_conditions else None

        # Perform a search for the top [max_results] nearest neighbors
        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector[0],
                limit=max_results,
                query_filter=filter_condition
            )

        except Exception as e:
            return f"搜索失败：{str(e)}"

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
    
    # the following function currently not provided to agents as a tool because it is a dangerous operation
    def delete_collection(
            self, 
            collection_name: str, 
        ):
        self.client.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")

    def show_collections(self):
        collections = self.client.get_collections()

        print(type(collections))
        print(f"Existing collections:")
        for collection in collections:
            print(f" - {collection}")

if __name__ == "__main__":
    qdrant = Qdrant_Handler()
    actions = ["create", "add", "search", "delete"]
    objects = ["collection", "data"]

    while True:
        print("Please select the action: ( create / add / search / delete )")
        selected_action = input()
        if selected_action.lower() == "exit":
            print("Exiting the program.")
            break
        if selected_action not in actions:
            print("Invalid action.")
            continue

        print("Please select the object: ( collection / data ) or 'back' to return to previous menu")
        selected_object = input()
        if selected_object.lower() == "back":
            continue
        if selected_object not in objects:
            print("Invalid object.")
            continue

        if selected_action == "create" and selected_object == "collection":
            print("Enter collection name (leave empty for default):")
            collection_name = input().strip()
            collection_name = None if collection_name == "" else collection_name
            qdrant.create_dataset(collection_name=collection_name)
            
        elif selected_action == "add" and selected_object == "data":
            print("Enter collection name (leave empty for default):")
            collection_name = input().strip()
            collection_name = None if collection_name == "" else collection_name
            
            print("Enter the message:")
            msg = input().strip()
            
            print("Enter the user role (e.g., user, assistant):")
            user_role = input().strip()
            
            qdrant.add_data(msg=msg, user_role=user_role, collection_name=collection_name)
            
        elif selected_action == "search" and selected_object == "data":
            print("Enter collection name (leave empty for default):")
            collection_name = input().strip()
            collection_name = None if collection_name == "" else collection_name
            
            print("Enter search query:")
            query = input().strip()
            
            print("Enter maximum number of results (leave empty for default):")
            max_results_str = input().strip()
            max_results = int(max_results_str) if max_results_str.isdigit() else 5
            
            print("Do you want to apply filters? (yes/no)")
            apply_filters = input().lower() == "yes"
            user_filter = None
            
            if apply_filters:
                user_filter = {}
                print("Enter speaker filter (leave empty to skip):")
                speaker = input().strip()
                if speaker:
                    user_filter["speaker"] = speaker
                
                print("Enter timestamp filter (format: YYYY-MM-DDTHH:MM:SS, leave empty to skip):")
                timestamp = input().strip()
                if timestamp:
                    user_filter["timestamp"] = timestamp
            
            result = qdrant.search_data(
                query=query, 
                user_filter=user_filter, 
                collection_name=collection_name,
                max_results=max_results
            )
            print("Search completed.")

        elif selected_action == "search" and selected_object == "collection":
            qdrant.show_collections()
            
        elif selected_action == "delete" and selected_object == "collection":
            print("Enter the name of the collection to delete:")
            collection_name = input().strip()
            
            print(f"Do you confirm to delete the collection '{collection_name}'? (yes/no)")
            confirm = input().lower()
            if confirm == "yes":
                try:
                    qdrant.delete_collection(collection_name=collection_name)
                    print(f"Collection '{collection_name}' deleted successfully.")
                except Exception as e:
                    print(f"Error deleting collection: {str(e)}")
            else:
                print("Operation cancelled.")

        else:
            print(f"Action '{selected_action}' on '{selected_object}' is not implemented yet.")
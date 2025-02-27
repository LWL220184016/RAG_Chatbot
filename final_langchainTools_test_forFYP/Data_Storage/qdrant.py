import os
import datetime

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

    def create_dataset(
            self, 
            collection_name: str, 
        ):

        if collection_name is None:
            collection_name = Database_Handler.get_newest_chat_name()

        if not self.client.collection_exists(collection_name):
            self.client.create_dataset(
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
            collection_name: str, 
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
            "id": self.pointID,
            "vector": embeddings[0].tolist(),
            "payload": {
                "msg": msg,
                **meta_data  # Merge additional JSON data into the payload
            }
        }

        # Insert (upsert) the points into the collection
        self.client.upsert(collection_name=collection_name, points=points)
        
        print(f"""Action: Qdrant_Handler add data ====================================")
            PointID: {self.pointID}
            Msg: {msg}
            Vector size: {embeddings.shape}
            Data inserted successfully.
            ===================================================================="""
        )

    def search_data(
            self, 
            query: str, 
            user_filter: Filter = None, 
            collection_name: str = None, 
        ):

        if collection_name is None:
            collection_name = Database_Handler.get_newest_chat_name()

        query_vector = self.embedder.embed([query])
        if user_filter.startswith("language="):
            language = user_filter.split("=")[1]
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="language",
                        match=MatchValue(value=language)
                    )
                ]
            )
        else:
            filter_condition = None

        # Perform a search for the top 2 nearest neighbors
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector[0],
            limit=2,
            query_filter=filter_condition
        )

        # Print out the search results
        print("Search results:")
        # for point in search_result:
        #     if isinstance(point, tuple):
        #         # If the result is a tuple, access elements by index
        #         point = point[1]
        #         for p in point:
        #             point_id = p.id
        #             similarity_score = p.score
        #             message = p.payload['msg']
        #             language = p.payload['language']
        return str(search_result)
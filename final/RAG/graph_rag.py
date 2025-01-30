# https://github.com/HKUDS/LightRAG
# for handle memory, can also consider run mem0 in Podman

# export NEO4J_URI="neo4j://localhost:7687"
# export NEO4J_USERNAME="neo4j"
# export NEO4J_PASSWORD="password"

import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
from lightrag.utils import EmbeddingFunc

class Graph_RAG:
    def __init__(
                self, 
                working_dir="./local_neo4j_storageDir", 
                llm_model_name='h2oai/h2o-danube3-500m-chat', 
                embedding_dim=384,
                max_token_size=5000,
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                # for neo4j storage
                graph_storage="Neo4JStorage", 
                log_level="DEBUG",
            ):
        """
        """
        
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        # Initialize LightRAG with Hugging Face model
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
            llm_model_name=llm_model_name,  # Model name from Hugging Face
            # Use Hugging Face embedding function
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=max_token_size,
                func=lambda texts: hf_embedding(
                    texts,
                    tokenizer=AutoTokenizer.from_pretrained(embedding_model_name),
                    embed_model=AutoModel.from_pretrained(embedding_model_name)
                )
            ),
            # for neo4j storage
            graph_storage=graph_storage, #<-----------override KG default
            log_level=log_level  #<-----------override log_level default
        )

    def load_file(self, file_dir):
        """
        file_dir: the file directory you want to load, example: "./book.txt"
        """
        with open(file_dir) as f:
            self.rag.insert(f.read())

    def search_rag(self, query, mode):
        """
        query: what you want to search in the RAG
        mode: naive, local, global, hybrid, mix
        """
        return self.rag.query(query, param=QueryParam(mode=mode))

        # Perform mix search (Knowledge Graph + Vector Retrieval)
        # Mix mode combines knowledge graph and vector search:
        # - Uses both structured (KG) and unstructured (vector) information
        # - Provides comprehensive answers by analyzing relationships and context
        # - Supports image content through HTML img tags
        # - Allows control over retrieval depth via top_k parameter

    def search_rag(self, query, prompt, mode):
        """
        query: What you want to search in the RAG
        prompt: What additional requirements do you want in the return
        mode: Naive, local, global, hybrid, mix
        """
        return self.rag.query_with_separate_keyword_extraction(
            query=query,
            prompt=prompt,
            param=QueryParam(mode=mode)
        )
        # self.rag.query_with_separate_keyword_extraction(
        #     query="Explain the law of gravity",
        #     prompt="Provide a detailed explanation suitable for high school students studying physics.",
        #     param=QueryParam(mode="hybrid")
        # )
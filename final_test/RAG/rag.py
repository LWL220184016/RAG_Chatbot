from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

class RAG:
    def __init__(
            self, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.current_chat = None
        try:
            self.faiss_index = FAISS.load_local("faiss_index", self.embedding_model)
        except ValueError:
            self.faiss_index = None

    def update_knowledge_base(self, documents):
        documents = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in documents]
        document_texts = [doc.page_content for doc in documents]
        # document_embeddings = embedding_model.embed_documents(document_texts) # may need this

        if self.faiss_index is None: # If no index exists, create one
            self.faiss_index = FAISS.from_documents(documents, self.embedding_model)

        else: # Add new documents to the existing index
            self.faiss_index.add_documents(documents)

        self.faiss_index.save_local("faiss_index")

    def search(self, llm, query):
        if self.faiss_index is None:
            return ""
        retrieval_qa = RetrievalQA.from_chain_type(
            llm = llm,
            retriever = self.faiss_index.as_retriever(),
            chain_type = "stuff"
        )
        response = retrieval_qa(query)
        return response

    def update_chat_history(self, user_message, llm_message):
        self.current_chat = [
            {"text": user_message.content, "metadata": {"timestamp": user_message.time, "sender": user_message.user_role}},
            {"text": llm_message.content, "metadata": {"timestamp": llm_message.time, "sender": llm_message.time}}
        ]
        return 
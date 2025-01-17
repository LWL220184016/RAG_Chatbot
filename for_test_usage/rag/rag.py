import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from llm_tts.llm import get_transformers_LLM

# Step 1: Prepare your documents
documents = [
    {
        "text": "這是一份有關氣候變化的報告，詳細討論了全球變暖的影響及其潛在解決方案。",
        "metadata": {"source": "Climate Research Journal, 2023"}
    },
    {
        "text": "根據最新的經濟數據，2023 年的全球經濟增長率預計將達到 4%。",
        "metadata": {"source": "Global Economic Outlook, 2023"}
    },
    {
        "text": "本研究探討了人工智慧在醫療領域的應用，包括診斷、治療和病人管理。",
        "metadata": {"source": "AI in Healthcare Review, 2023"}
    },
    {
        "text": "在2022年，數字化轉型成為企業成功的關鍵，尤其是在零售和金融行業。",
        "metadata": {"source": "Digital Transformation Insights, 2022"}
    },
    {
        "text": "這篇文章分析了社交媒體對青少年心理健康的影響，並提出了相應的建議。",
        "metadata": {"source": "Journal of Adolescent Health, 2023"}
    },
    {
        "text": "根據調查，90%的企業認為可持續性是未來商業策略的重要組成部分。",
        "metadata": {"source": "Sustainability in Business, 2023"}
    },
    {
        "text": "本報告提供了2023年全球科技趨勢的概述，包括區塊鏈和物聯網的發展。",
        "metadata": {"source": "Tech Trends Report, 2023"}
    },
    {
        "text": "研究顯示，定期運動能顯著改善心理健康和整體福祉。",
        "metadata": {"source": "Health and Wellness Journal, 2023"}
    },
    {
        "text": "本文件探討了新興市場的投資機會，特別是在亞洲地區。",
        "metadata": {"source": "Emerging Markets Investment Report, 2023"}
    },
    {
        "text": "這篇文章討論了全球能源轉型的挑戰和機遇，特別是在可再生能源方面。",
        "metadata": {"source": "Energy Transition Journal, 2023"}
    }
]


# Convert dictionaries to Document objects
documents = [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in documents]

# Step 2: Create embeddings for your documents
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
document_texts = [doc.page_content for doc in documents]
document_embeddings = embedding_model.embed_documents(document_texts)

# Step 3: Index the embeddings with FAISS
faiss_index = FAISS.from_documents(documents, embedding_model)
faiss_index.save_local("faiss_index")

# Step 4: Integrate with LangChain
llm = get_transformers_LLM("h2oai/h2o-danube3-500m-chat")
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_index.as_retriever(),
    chain_type="stuff"
)

# Example query
query = "please tell what happening you know in 2022"
response = retrieval_qa(query)
print()
print()
print()

print(response)


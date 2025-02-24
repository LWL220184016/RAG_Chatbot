# import os

# from langchain_core.memory import BaseMemory
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationChain
# from mem0 import MemoryClient

# openai_api_key = os.environ["OPENAI_API_KEY"]
# config = {
#     "vector_store": {
#         "provider": "qdrant",
#         "config": {
#             "host": "localhost",
#             "port": 6333,
#         },
#     },
#     "llm": {
#         "provider": "openai",
#         "config": {
#             "model": "gpt-3.5-turbo",
#             "api_key": openai_api_key,
#         },
#     },
#     "embedder": {
#         "provider": "openai",
#         "config": {
#             "model": "text-embedding-3-large",
#             "api_key": openai_api_key,
#         },
#     },
# }

# mem0_client = MemoryClient(config)

# class Mem0Memory(BaseMemory):
#     def __init__(self, mem0_client, user_id):
#         self.mem0_client = mem0_client
#         self.user_id = user_id

#     def save_context(self, inputs, outputs):
#         user_message = {"role": "user", "content": inputs}
#         assistant_message = {"role": "assistant", "content": outputs}
#         messages = [user_message, assistant_message]
#         self.mem0_client.add(messages, user_id=self.user_id)

#     def load_memory_variables(self, inputs):
#         query = inputs
#         related_memories = self.mem0_client.search(query, user_id=self.user_id)
#         memories_content = [memory['memory'] for memory in related_memories]
#         history = '\n'.join(memories_content)
#         return {'history': history}

# user_id = "user1"
# memory = Mem0Memory(mem0_client, user_id)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
# conversation = ConversationChain(llm=llm, memory=memory)

# response = conversation.predict(input="Hello, how are you?")
# print(response)
# response = conversation.predict(input="What's the weather like today?")
# print(response)

import os
from langchain_core.memory import BaseMemory
from langchain.chains import ConversationChain
from mem0 import MemoryClient
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

# Mem0 配置，使用 Transformers 模型
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        },
    },
    "llm": {
        "provider": "custom",  # 自訂提供者，使用本地 Transformers 模型
        "config": {
            "model": "distilgpt2",  # 使用輕量化的 distilgpt2 模型
        },
    },
    "embedder": {
        "provider": "custom",  # 自訂嵌入模型
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",  # 用於生成嵌入的模型
        },
    },
}

mem0_client = MemoryClient(config)

# 自訂 LLM 類別，適配 LangChain
class TransformersLLM:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # 設置為評估模式

    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 自訂嵌入類別
class TransformersEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用最後一層的平均池化作為嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

# Mem0 記憶類別
class Mem0Memory(BaseMemory):
    def __init__(self, mem0_client, user_id):
        self.mem0_client = mem0_client
        self.user_id = user_id

    def save_context(self, inputs, outputs):
        user_message = {"role": "user", "content": inputs}
        assistant_message = {"role": "assistant", "content": outputs}
        messages = [user_message, assistant_message]
        self.mem0_client.add(messages, user_id=self.user_id)

    def load_memory_variables(self, inputs):
        query = inputs
        related_memories = self.mem0_client.search(query, user_id=self.user_id)
        memories_content = [memory['memory'] for memory in related_memories]
        history = '\n'.join(memories_content)
        return {'history': history}

# 初始化
user_id = "user1"
memory = Mem0Memory(mem0_client, user_id)
llm = TransformersLLM(model_name="distilgpt2")

# 將 Transformers LLM 包裝為 LangChain 的 ConversationChain 可用格式
class WrappedLLM:
    def __init__(self, llm):
        self.llm = llm

    def predict(self, input, history=""):
        prompt = f"{history}\nHuman: {input}\nAssistant: "
        return self.llm(prompt)

wrapped_llm = WrappedLLM(llm)
conversation = ConversationChain(llm=wrapped_llm, memory=memory)

# 測試對話
response = conversation.predict(input="Hello, how are you?")
print(response)
response = conversation.predict(input="What's the weather like today?")
print(response)
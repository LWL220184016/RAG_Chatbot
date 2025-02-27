import os
import torch
from langchain_core.memory import BaseMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextStreamer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance


# 本地 Qdrant 客户端类
class LocalQdrantClient:
    def __init__(self, host="localhost", port=6333, collection_name="memories"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.client.recreate_dataset(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def add(self, messages, user_id):
        points = [
            PointStruct(
                id=f"{user_id}_{i}",
                vector=self.embed(message["content"]),
                payload={"role": message["role"], "content": message["content"], "user_id": user_id}
            )
            for i, message in enumerate(messages)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query, user_id):
        query_vector = self.embed(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter={
                "must": [
                    {
                        "key": "user_id",
                        "match": {"value": user_id}
                    }
                ]
            },
            limit=10
        )
        return [{"memory": hit.payload["content"]} for hit in search_result]

    def embed(self, text):
        # 使用嵌入模型生成嵌入向量
        embedder = TransformersEmbedder()
        return embedder.embed(text)

class TransformersLLM(BaseLLM):
    model_name: str = "distilgpt2"
    tokenizer: Any = None
    model: Any = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",  # This helps with device placement
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    @property
    def _llm_type(self) -> str:
        return "custom_transformers"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Ensure inputs are on the correct device
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate completions for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

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

# 修改後的 Mem0Memory 類別，明確宣告 client 與 user_id
class Mem0Memory(BaseMemory):
    client: object
    user_id: str

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def memory_variables(self) -> list[str]:
        return ["history"]

    def load_memory_variables(self, inputs: dict) -> dict:
        # Extract the actual query from inputs
        if isinstance(inputs, dict):
            query = inputs.get("input", "")
        else:
            query = str(inputs)

        # Ensure query is a string
        query = str(query)
        
        related_memories = self.client.search(query, user_id=self.user_id)
        memories_content = [memory['memory'] for memory in related_memories]
        history = '\n'.join(memories_content)
        return {'history': history}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        # Extract actual input/output content
        input_str = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
        output_str = outputs.get("output", "") if isinstance(outputs, dict) else str(outputs)
        
        user_message = {"role": "user", "content": input_str}
        assistant_message = {"role": "assistant", "content": output_str}
        messages = [user_message, assistant_message]
        self.client.add(messages, user_id=self.user_id)

    def clear(self) -> None:
        pass

# 初始化
user_id = "user1"
local_qdrant_client = LocalQdrantClient()
memory = Mem0Memory(client=local_qdrant_client, user_id=user_id)
llm = TransformersLLM(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

# Define your tools (if any)
def search_ddg(query: str) -> str:
    """Search DuckDuckGo."""
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    return search.run(query)

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Create tools
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_ddg,
        description="Useful for searching the internet for current information",
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Useful for performing mathematical calculations",
    )
]

agent = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors="Check your output format!",
)

# Test the agent
if __name__ == "__main__":
    print("\nTesting Search Tool:")
    response = agent.run("What is the current weather in Tokyo?")
    print("Search response:", response)

    print("\nTesting Calculator Tool:")
    response = agent.run("What is 123 * 456?")
    print("Calculator response:", response)

    print("\nTesting Memory:")
    response = agent.run("What did we just calculate?")
    print("Memory response:", response)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import multiprocessing
from mem0 import Memory
from final_langchainTools_forFYP.LLM.llm_google import LLM_Google as LLM


# Change the embedding model to one with a known dimension
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "memory_collection",
        }
    },
    # "graph_store": {
    #     "provider": "neo4j",
    #     "config": {
    #         "url": "bolt://localhost:7687",
    #         "username": "neo4j",
    #         "password": "StrongPassword123"
    #     },
    #     "custom_prompt": "Please only extract entities containing sports related relationships and nothing else.",
    # },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-1.5-flash-latest",
            "temperature": 0.2,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja",  # This has 384 dimensions
        }
    },
    "api_version": "v1.1",
}

# Create a clean memory instance with API version 1.1
memory = Memory.from_config(config_dict=config)

is_user_talking = multiprocessing.Event()
stop_event = multiprocessing.Event()
speaking_event = multiprocessing.Event()

llm = LLM(
    model_name="gemini-1.5-flash-latest",
    is_user_talking=is_user_talking,
    stop_event=stop_event,
    speaking_event=speaking_event,
    tools=[],
)

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    memory.add(f"User asked: {message}", user_id=user_id, metadata={"category": "User_question"})
    
    # Retrieve relevant memories with API v1.1
    search_results = memory.search(query=message, user_id=user_id, limit=5)
    
    # Process search results for v1.1 format - should be a list of Memory objects
    relevant_memories = search_results
    
    print(f"Found {len(relevant_memories)} relevant memories")
    
    # Format memories for the prompt - handling v1.1 API format
    if relevant_memories:
        # v1.1 API returns Memory objects with 'content' attribute
        memories_str = "\n".join(f"- {memory_obj.content}" for memory_obj in relevant_memories)
    else:
        memories_str = "No relevant memories found."
    
    # Generate Assistant response with specific context instructions
    system_prompt = (
        f"You are a helpful AI assistant with memory of past conversations. "
        f"Please answer the user's question directly.\n\n"
        f"Context from previous conversations:\n{memories_str}\n\n"
        f"User's current question: {message}\n\n"
        f"Answer based on both the current question and any relevant context from memory."
    )
    
    response = llm.agent.invoke(system_prompt)  # Updated from deprecated run() method
    assistant_response = response
    
    # Store the response in memory with API v1.1
    memory.add(f"AI responded: {assistant_response}", user_id=user_id, metadata={"category": "AI_response"})
    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest", top_k=10, top_p=0.95, temperature=0.8)

for txt in llm.stream("What is the capital of France?"):
    print(txt)
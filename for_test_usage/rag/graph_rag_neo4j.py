# https://github.com/HKUDS/LightRAG
# curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
# for handle memory, can also consider run mem0 in Podman

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from final.RAG.graph_rag import Graph_RAG

WORKING_DIR = "./local_neo4j_storageDir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize LightRAG with Hugging Face model
rag = Graph_RAG(working_dir=WORKING_DIR)

rag.load_file("./book.txt")

# # mode: naive, local, global, hybrid, mix
# result = rag.search_rag_noPrompt("What are the top themes in this story?", "mix")
# print(result)


result = rag.search_rag_noPrompt("What is the names of characters in this story?", "naive")
print(result)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from final.Data_Storage.neo4j import Neo4J
from final.LLM.prompt_template import Message


if __name__ == "__main__":
    neo4j = Neo4J()
    print("Neo4J instance created")
    user_message = Message("best friend1")
    llm_message = Message("best friend2")
    # user_message.update_content("Hello, what's up?")
    # llm_message.update_content("Hello, I'm fine, thank you.")
    user_message.update_content("What is the weather like today?")
    llm_message.update_content("The weather is sunny today.")

    neo4j.add_dialogue_record(user_message, llm_message)
    print("Neo4J record added")
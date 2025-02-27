import os

from neo4j import GraphDatabase
from datetime import datetime
from database import Database_Handler

class Neo4J(Database_Handler):
    def __init__(
            self, 
            dataID: int = None
        ):

        """
        You should run the following commands in the terminal before running the code:

        URL examples: 
            export NEO4J_URI="neo4j://localhost" or "neo4j+s://xxx.databases.neo4j.io" 
        AUTH examples: 
            export NEO4J_USERNAME="username"
            export NEO4J_PASSWORD="password"
        """
        super(dataID)
        self.driver = GraphDatabase.driver(
            uri = os.getenv('NEO4J_URI'), 
            auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')), 
        )
        self.create_dataset()

    def create_dataset(
            self, 
            node_name: str, 
        ):

        if node_name is None:
            node_name = super().get_newest_chat_name()

        print(f"create Chat {node_name} node in Neo4J if not exist", node_name)
        # MERGE checks for the existence of a node with the specified properties 
        # and creates it only if it does not already exist
        with self.driver.session() as session:
            session.run(
                "MERGE (c:Chat {date: $node_name}) "
                "MERGE (d:Chat ) "
                "MERGE (c)-[:SPEAKS]->(d)",
                node_name=node_name
            )
            # # If the above not work
            # session.run(
            #     "MERGE (c:Chat {date: $node_name}) "
            #     "MERGE (d:Chat {content: $content, timestamp: $timestamp, speaker: $speaker}) "
            #     "MERGE (c)-[:SPEAKS]->(d)",
            #     node_name=node_name, content=content, timestamp=timestamp, speaker=speaker
            # )
    
    def add_data(
            self, 
            msg, 
            user_role, 
            node_name: str, 
        ):
        """
        user_message & llm_message:
            Class Message in LLM.prompt_template
        """
        
        self.dataID += 1
        if node_name is None:
            node_name = super().get_newest_chat_name()

        time = datetime.now().isoformat()
        with self.driver.session() as session:
            # 合并用户消息和AI消息处理
            session.run(
                """
                MERGE (c:Chat {date: $node_name})
                MERGE (d:Chat {
                    text: $chat_text, 
                    timestamp: $timestamp, 
                    speaker: $speaker
                })
                MERGE (c)-[:SPEAKS]->(d)
                """,
                node_name=node_name,
                chat_text=msg,
                timestamp=time,
                speaker=user_role,
            )
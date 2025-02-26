import os
from neo4j import GraphDatabase
from datetime import datetime

class Neo4J():
    def __init__(self):
        """
        You should run the following commands in the terminal before running the code:

        URL examples: 
            export NEO4J_URI="neo4j://localhost" or "neo4j+s://xxx.databases.neo4j.io" 
        AUTH examples: 
            export NEO4J_USERNAME="username"
            export NEO4J_PASSWORD="password"
        """
        self.URL = os.getenv('NEO4J_URI')
        self.AUTH = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        self.driver = self.start_driver()
        self.create_dialogue_node()

    def start_driver(self, ):
        driver = GraphDatabase.driver(uri=self.URL, auth=self.AUTH)
        return driver

    def create_dialogue_node(self, ): 
        today = datetime.today().strftime('%Y-%m-%d')
        print(f"create Chat {today} node in Neo4J if not exist", today)
        # MERGE checks for the existence of a node with the specified properties 
        # and creates it only if it does not already exist
        with self.driver.session() as session:
            session.run(
                "MERGE (c:Chat {date: $today}) "
                "MERGE (d:Dialogue ) "
                "MERGE (c)-[:SPEAKS]->(d)",
                today=today
            )
            # # If the above not work
            # session.run(
            #     "MERGE (c:Chat {date: $today}) "
            #     "MERGE (d:Dialogue {content: $content, timestamp: $timestamp, speaker: $speaker}) "
            #     "MERGE (c)-[:SPEAKS]->(d)",
            #     today=today, content=content, timestamp=timestamp, speaker=speaker
            # )
    
    def add_dialogue_record(
            self, 
            user_message, 
            llm_message, 
        ):
        """
        user_message & llm_message:
            Class Message in LLM.prompt_template
        """

        today = datetime.today().strftime('%Y-%m-%d')
        
        with self.driver.session() as session:
            # 合并用户消息和AI消息处理
            for message in [user_message, llm_message]:
                session.run(
                    """
                    MERGE (c:Chat {date: $today})
                    MERGE (d:Dialogue {
                        text: $dialogue_text, 
                        timestamp: $timestamp, 
                        speaker: $speaker
                    })
                    MERGE (c)-[:SPEAKS]->(d)
                    """,
                    today=today,
                    dialogue_text=message,
                    timestamp=message.time,
                    speaker=message.user_role,
                )
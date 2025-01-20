from neo4j import GraphDatabase
from datetime import datetime

class Neo4J():
    def __init__(self):
        # URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
        self.URI = "<URI for Neo4j database>"
        self.AUTH = ("<Username>", "<Password>")
        self.driver = self.start_driver()
        self.create_dialogue_node()

    def start_driver(self, ):
        driver = GraphDatabase.driver(self.URI, auth=self.AUTH)
        print(str(driver.verify_connectivity()))
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
    
    def add_dialogue_record(self, today, dialogue_text, timestamp, speaker):
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Dialogue {text: $dialogue_text, timestamp: $timestamp, speaker: $speaker}) "
                "MATCH (c:Chat {date: $today}) "
                "MERGE (c)-[:SPEAKS]->(d)",
                today=today,
                dialogue_text=dialogue_text,
                timestamp=timestamp,
                speaker=speaker,
            )
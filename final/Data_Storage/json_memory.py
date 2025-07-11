import json
import datetime

from Data_Storage.database import Database_Handler

class JSON_Memory:

    def __init__(self, task: str = "chat_history_record", path: str = None):
        """
        tasks:
        1. chat_history_record
        2. temp_memory
        """

        if path:
            CURRENT_DIR = path
        else:
            # Use the directory of this file as the default path
            import os
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            
        if task == "chat_history_record":
            collection_name = Database_Handler.get_newest_chat_name()
            self.json_file_path = CURRENT_DIR + "/chat_history/" + collection_name + ".json" 
        elif task == "temp_memory":
            self.json_file_path = CURRENT_DIR + "/chat_history/temp.json"  # Removed trailing comma
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.json_file_path), exist_ok=True)
        
        if os.path.exists(self.json_file_path):
            print("Loading the json memory file")
            self.memory = self.load(self.json_file_path)
        else:
            print("The json memory file does not exist. Creating new file.")
            self.memory = {"chat_records": []}  # Direct dictionary instead of json.loads
            with open(self.json_file_path, "w") as f:
                json.dump(self.memory, f)

    def get(self):
        print("Getting the json memory")
        return self.memory

    def add(self, user_message: str, llm_message: str, chat_record_limit: int = 20):
        """
        Add a message to the chat records.
        
        Args:
            role: The role of the sender (e.g., 'user', 'assistant')
            message: The message content
            chat_record_limit: Optional limit to the number of records to keep
        """
        self.memory["chat_records"].append({
            "user_message": user_message, 
            "llm_message": llm_message, 
            "timestamp": str(datetime.datetime.now())
        })
        
        if chat_record_limit and len(self.memory["chat_records"]) > chat_record_limit:
            self.memory["chat_records"].pop(0)

        self.save(self.json_file_path)

    def add_no_limit(self, user_message: str, llm_message: str):
        """
        Add a message to the chat records.
        
        Args:
            role: The role of the sender (e.g., 'user', 'assistant')
            message: The message content
        """
        self.memory["chat_records"].append({
            "user_message": user_message, 
            "llm_message": llm_message, 
            "timestamp": str(datetime.datetime.now())
        })
        
        self.save(self.json_file_path)

    def save(self, file_path):
        try:
            with open(file_path, 'w') as f:
                json.dump(self.memory, f)
        except Exception as e:
            print(f"Error saving memory to {file_path}: {e}")

    def load(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading memory from {file_path}: {e}")
            return {"chat_records": []}
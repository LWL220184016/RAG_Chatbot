import json
import os
import datetime

class JSON_Temp_Memory:
    def __init__(self, json_file_path: str = "", chat_record_limit: int = 10):
        if os.path.exists(json_file_path):
            print("loading the json temp memory file")
            self.temp_memory = self.load(json_file_path)
        else:
            print("The json temp memory file does not exist.")
            self.temp_memory = json.loads("{\"chat_records\": []}")

        self.json_file_path = json_file_path
        self.chat_record_limit = chat_record_limit # 1 user message + 1 agent message = 1 chat record

    def get(self):
        return self.temp_memory

    def add(self, role: str, message: dict):
        self.temp_memory["chat_records"].append({"role": role, "message": message, "timestamp": str(datetime.datetime.now())})
        if len(self.temp_memory["chat_records"]) > self.chat_record_limit:
            self.temp_memory["chat_records"].pop(0)

        # self.save(self.json_file_path)

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.temp_memory, f)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            self.temp_memory = json.load(f)
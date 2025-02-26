import json
import time

class Message():
    def __init__(
            self, 
            user_role
        ):
        
        self.user_role = user_role
        self.content = None
        self.mood = None
        self.emoji = None
        self.memory = None
        self.time = None

    def update_content(self, content, mood=None, emoji=None, memory=None):
        self.content = content
        self.mood = mood
        self.emoji = emoji
        self.memory = memory
        self.time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))

        return json.dumps(self.__dict__)

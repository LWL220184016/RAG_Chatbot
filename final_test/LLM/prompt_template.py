import json
import time

class Message():
    def __init__(self, user_role):
        self.user_role = user_role
        self.content
        self.mood
        self.emoji
        self.time

    def update_content(self, content, mood=None, emoji=None):
        self.content = content
        self.mood = mood
        self.emoji = emoji
        self.time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))

        return json.dumps(self.__dict__)

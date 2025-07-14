import json
import time

class Message():
    def __init__(
            self, 
            user_role
        ):
        
        self.user_role = user_role
        self.system_msg = "You are a good friend with the user. " \
                          "You need to communicate with him like a best friend. " \
                          "You should only output spoken sentences with emoji to show your current mood, not formatted content. " \
                          "You should only output Chinese or English. " \
                          "When he is silent, attribute named content will be blank in the message and you can also remain silent or actively seek topics to talk about. " \
                          "You should also consider the attribute named time in the message you receive, " \
                          "If it's too close to the last time you spoke, you should keep silent. " 

    def update_content(self, content, mood=None, emoji=None, memory=None):

        # messages = [
        #     {
        #         "role": self.user_role, 
        #         "content": content, 
        #         "mood": mood, 
        #         "emoji": emoji, 
        #         "memory": memory, 
        #         "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())), 
        #         "system_msg": self.system_msg
        #     }
        # ]
        
        timestamp = f" [Timestamp: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))}] "
        messages = [
            {"role": self.user_role, "content": content + "" if mood == None else f" [user_mood: {mood}] " }, 
            {"role": "system", "content": self.system_msg + timestamp + f"[History: {memory}] " }, 
        ]

        # 把 timestamp 分開是為了取得更精確的時間戳
        return messages, timestamp

if __name__ == "__main__":
    # 測試 Message 類別
    msg = Message(user_role="user")
    content = "Hello, how are you?"
    prompt, timestamp = msg.update_content(content=content, memory=None)
    prompt[1]["content"] = "You are agent1. " + prompt[1]["content"]

    print(f"Prompt: {prompt}")
    print(f"Timestamp: {timestamp}")

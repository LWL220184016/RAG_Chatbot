import json
import time
from langchain.prompts import PromptTemplate

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

def get_langchain_PromptTemplate():
    """<|IS|>: 用于标记来源信息的链接，避免 URL 进入 TTS """

    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="""
            用户问题：{user_input}
            根据用户的问题，通过网络搜索验证信息，并嚴格按照下方指定模板输出最终答案，包含相关信息来源链接.
            使用和用户相同的語言或者用户要求的语言進行回答，以及必须在 "信息来源：" 前面加上特殊标记 <|IS|>。

            若用户要求推荐食谱，请按以下格式输出结果(列点不应该和模板一样限定为两个，应该展示你所有认证为正确的信息)：  
            食谱名称：[食谱的名称]  

            制作步骤：  
                1. ...
                2. ...

            食材名称及分量：  
                1. ...
                2. ...

            制作时间：[制作时间]  
            推荐原因：[推荐的原因]  
            信息来源：[相关链接]

            若用户要求提供食物信息，请按以下格式输出结果(列点不应该和模板一样限定为两个，应该展示你所有认证为正确的信息)：  
            食物名称：[食物的名称]  
            食物的其他名稱(包含它的別名, 其他語言的叫法)：
                1. ...
                2. ...

            营养成分：  
                1. ...
                2. ...

            原产地：[原产地]  
            品种：  
                1. ...
                2. ...

            可能存在的问题：  
                1. ...
                2. ...

            总结：[总结]
            
            <|IS|>信息来源：[相关链接]

            注意：  
            列点数量应根据验证后的信息灵活调整，不限于模板中的示例数量。  
            确保所有信息经过验证，来源可靠，并在输出中提供链接。
        """,
    )
    return prompt_template
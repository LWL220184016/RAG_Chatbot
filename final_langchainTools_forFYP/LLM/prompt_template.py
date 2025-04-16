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
        self.system_msg = "Your current role is a good friend with the user. You need to communicate with him like a best friend. When he is silent, the system will send [User did not speak] to indicate that he has not spoken for a while, you can also remain silent or actively seek topics to talk about."

    def update_content(self, content, mood=None, emoji=None, memory=None):
        self.content = content
        self.mood = mood
        self.emoji = emoji
        self.memory = memory
        self.time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))

        return json.dumps(self.__dict__)

def get_langchain_PromptTemplate_Chinese1():
    """<|IS|>: 用于标记来源信息的链接，避免 URL 进入 TTS """

    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="""
            用户问题：{user_input}
            分析用户的问题，拒绝回答一切和食物无关的问题。
            如果是食物相关的问题，首先通过网络搜索验证信息，并嚴格按照下方指定模板输出最终答案，包含相关信息来源链接.
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

def get_langchain_PromptTemplate_Chinese2():
    """<|IS|>: 用于标记来源信息的链接，避免 URL 进入 TTS """

    prompt_template = PromptTemplate(
        input_variables=["user_input", "memory"], # 用戶輸入和歷史對話要點
        template="""
            用户问题：{user_input}

            历史对话要点：{memory} 

            **任务目标：** 分析用户的问题以及历史对话要点，**仅回答**食物相关的问题。如果问题与食物无关，请**拒绝回答**。

            **回答流程：**

            1. **问题分析 (Thought):**  仔细分析用户提出的问题，判断是否与食物相关。
                * 如果问题与食物无关，直接拒绝回答。
                * 如果问题与食物相关，进入下一步。

            2. **网络搜索 (Action & Observation):**  使用网络搜索工具验证信息，获取回答用户问题所需的资料。
                *  根据问题类型选择合适的关键词进行搜索。
                *  仔细阅读搜索结果 (Observation)，提取关键信息，验证信息的可靠性。

            3. **答案生成 (Final Answer):**  **基于验证后的信息，严格按照下方指定的输出模板输出最终答案。**
                *  使用与用户相同的语言或用户要求的语言进行回答。
                *  必须在結束思考以及開始输出最終答案前加上特殊标记 </think>。
                *  必须在输出模板中的 "信息来源：" 前面加上特殊标记 <|IS|>。
                *  列点数量应根据验证后的信息灵活调整，不限于输出模板中的示例数量。
                *  确保所有信息经过验证，来源可靠，并在输出中提供链接。

            **输出模板：**

            **若用户要求推荐食谱，请按以下格式输出结果：**

            食谱名称：[食谱的名称]

            制作步骤：
                1. ...
                2. ...
                ... (根据实际步骤数量列出)

            食材名称及分量：
                1. ...
                2. ...
                ... (根据实际食材数量列出)

            制作时间：[制作时间]
            推荐原因：[推荐的原因]
            <|IS|>信息来源：[相关链接]
            ... (根据实际链接数量列出)


            **若用户要求提供食物信息，请按以下格式输出结果：**

            食物名称：[食物的名称]
            食物的其他名稱(包含它的別名, 其他語言的叫法)：
                1. ...
                2. ...
                ... (根据实际别名数量列出)

            营养成分：
                1. ...
                2. ...
                ... (根据实际营养成分数量列出)

            原产地：[原产地]
            品种：
                1. ...
                2. ...
                ... (根据实际品种数量列出)

            可能存在的问题：
                1. ...
                2. ...
                ... (根据实际问题数量列出)

            总结：[总结]
            <|IS|>信息来源：[相关链接]
            ... (根据实际链接数量列出)


            **特别注意：**

            *   **严格遵守输出模板。**
            *   **信息必须经过网络搜索验证，确保准确可靠。**
            *   **答案必须包含信息来源链接，并使用 <|IS|> 特殊标记。**
            *   **列点数量应根据验证后的信息灵活调整，不限于模板中的示例数量。**
            *   **使用和用户相同的语言或者用户要求的语言進行回答。**

            **修改说明：**

            1.  **更清晰的任务目标和流程：**  在提示詞的開頭，我添加了 "**任务目标**" 和 "**回答流程**" 部分，更清晰地描述了 Agent 的任務和執行步驟，**強調 Agent 需要完成從問題分析到最終答案生成的完整流程**。
            2.  **強化 "答案生成 (Final Answer)" 指令：**  在 "**回答流程**" 中，我明確指出在 "**答案生成 (Final Answer)**" 階段，Agent 需要 "**基于验证后的信息，严格按照下方指定的模板输出最终答案**"。  **這一步驟旨在引導 Agent 在 Observation 後，必須進行答案生成，而不是停留在 Observation 階段。**
            3.  **強調信息驗證和來源追溯：**  在 "**回答流程**" 和 "**特别注意**" 部分，我多次強調 "**信息必须经过网络搜索验证，确保准确可靠**" 和 "**答案必须包含信息来源链接**"，**確保 Agent 生成的答案是基於可靠信息來源的，並符合您的要求。**
            4.  **保留原有格式和語言要求：**  修改後的提示詞模板完全保留了您原有的輸出格式、語言要求和特殊標記 `<|IS|>`，確保修改後的提示詞仍然符合您的具體需求。

            **使用建議：**

            1.  **替換您 Agent 中原有的提示詞模板。**
            2.  **使用修改後的提示詞模板重新測試您的 Agent。**  觀察 Agent 是否能夠穩定地完成 「Thought → Action → Observation → Final Answer」 循環，並解決之前偶爾中斷的問題。
            3.  **如果問題仍然存在，請檢查 Agent 的代碼實現。**  提示詞模板的修改主要在指令 Agent 的行為，如果問題仍然存在，可能需要檢查 Agent 的代碼實現，例如 Agent 的工具調用、Observation 解析、答案生成等環節是否存在 Bug。
        """,
    )
    return prompt_template

def get_langchain_PromptTemplate_English1():
    """<|IS|>: 用于标记来源信息的链接，避免 URL 进入 TTS """

    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="""
            User question: {user_input}
            Analyze the user's question and refuse to answer any questions not related to food.
            If it is a food-related question, first verify the information through online search, and strictly output the final answer according to the specified template below, including the link to the relevant information source.
            Answer in the same language as the user or the language requested by the user, and must add a special mark <|IS|> before "Information source:"

            If the user asks for recipe recommendations, please output the results in the following format (the list should not be limited to two like the template, and should show all the information you have verified to be correct):
            Recipe name: [recipe name]

            Preparation steps:
            1. ...
            2. ...

            Ingredients and quantities:
            1. ...
            2. ...

            Preparation time: [preparation time]
            Recommendation reason: [recommendation reason]
            Information source: [related link]

            If the user asks for food information, please output the results in the following format (the list should not be limited to two like the template, and should show all the information you have verified to be correct):
            Food name: [food name]
            Other names of the food (including its aliases, names in other languages):
            1. ...
            2. ...

            Nutritional content:
            1. ...
            2. ...

            Origin: [origin]
            Variety:
            1. ...
            2. ...

            Possible problems:
            1. ...
            2. ...

            Summary: [Summary]

            <|IS|> Information Source: [Related Links]

            Note:
            The number of columns should be adjusted according to the verified information and is not limited to the number of examples in the template.
            Make sure all information is verified and reliable, and provide links in the output.
        """,
    )
    return prompt_template

def get_non_agent_langchain_PromptTemplate_chinese():
    """<|IS|>: 用于标记来源信息的链接，避免 URL 进入 TTS """

    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="""
            用户问题：{user_input}
            工具呼叫結果：{tool_output}

            分析用户的问题，拒绝回答和食物无关的问题。
            如果是食物相关的问题，则严格按照下方步骤实用工具来通过网络搜索验证信息，
            并嚴格按照下方指定模板输出最终答案，包含相关信息来源链接。
            使用和用户相同的語言或者用户要求的语言進行回答，以及必须在 "信息来源：" 前面加上特殊标记 <|IS|>。

            可以使用的工具：
                duckduckgo_searching：
                    "Tool_Name": "duckduckgo_searching"
                    "args": {"query": "[從用戶問題中提取的關鍵信息]"}
                    工具介紹：使用 DuckDuckGo 搜索引擎獲取網絡上的最新信息，工具的参数应该是一个字典，包含键值对，键是工具的参数名，值是参数的值。

            工具呼叫模板：
            {
                "Tool_Name": "[工具的名称]",
                "args": "[工具的参数]",
            }

            推理步骤（使用類似 ReAct 的方式）：
                1. 分析问题，提取关键信息然後輸出 json 格式數據呼叫工具并且結束推理。
                2. 如果工具呼叫結果不爲空，則分析并验证信息的可靠性，如果信息有相关性不足，不可靠等问题，重复步骤 1，如果信息可靠，則按照下方输出模板输出最终答案，包含相关信息来源链接。

            输出模板：
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
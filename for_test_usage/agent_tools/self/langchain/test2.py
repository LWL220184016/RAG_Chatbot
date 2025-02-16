# 安装必要库
# pip install langchain-community duckduckgo-search tenacity

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import Tool
from langchain_ollama import OllamaLLM
from tenacity import retry, stop_after_attempt, wait_fixed

from tools.duckduckgo_searching import duckduckgo_search
from ollamaStreamingCallbackHandler import OllamaStreamingCallbackHandler

# 初始化模型（降低随机性）
llm = OllamaLLM(
    # model="llama3-groq-tool-use",
    # model="deepseek-r1_14b_FYP4",
    # model="llama3.2:1b",
    # model="deepseek-r1:32b",
    model="deepseek-r1:14b",
    temperature=0.0,
    top_p=0.9,
    streaming=True,  # 启用流式传输
    callbacks=[StreamingStdOutCallbackHandler()],  # 标准输出回调
)

# 加载基础工具（添加错误处理）
# tools = load_tools(["ddg-search", "llm-math"], llm=llm)
tools = []
# 自定义工具（带重试和错误处理）
@tool
def get_current_time() -> str:
    """Returns current local time in HH:MM AM/PM format. Input is always empty."""
    try:
        import datetime
        return datetime.datetime.now().strftime("%I:%M %p")
    except Exception as e:
        return f"[Error] Time fetch failed: {e}"

# tools.append(
#     Tool(
#         name="current_time",
#         func=get_current_time,
#         description="Essential for queries about CURRENT LOCAL TIME. Input is empty."
#     )
# )
tools.append(get_current_time)
tools.append(duckduckgo_search)

# 创建Agent（结构化类型）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Check your output format!",
    callbacks=[OllamaStreamingCallbackHandler()],  # 绑定自定义回调
)

# system_message = """系统消息：根据用户的问题通过网络搜索验证信息并以下方模板在最终答案里面通过显示相关信息的来源链接。
#     如果用户要求推荐食谱，你需要通过以下格式来输出结果(列点不应该和模板一样限定为两个，应该展示你所有认证为正确的信息)：
#       食谱名称：[食谱的名称]\n
#       制作步骤：
#           1. ...\n
#           2. ...\n
#       食材名称以及分量：\n
#           1. ...\n
#           2. ...\n
#       制作时间：[制作时间]\n
#       推荐原因：[推荐的原因]\n\n
#     如果用户要求的是食物的信息，你需要通过以下格式来输出结果(列点不应该和模板一样限定为两个，应该展示你所有认证为正确的信息)：
#       食物名称：[食物的名称]\n
#       营养成分：\n
#           1. ...\n
#           2. ...\n
#       原产地：[原产地]\n
#       品种：\n
#           1. ...\n
#           2. ...\n
#       可能存在的问题：\n
#           1. ...\n
#           2. ...\n
#     ；
# """


prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""
        用户问题：{user_input}
        根据用户的问题，通过网络搜索验证信息，并按照下方指定模板输出最终答案，包含相关信息来源链接。  

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

        信息来源：[相关链接]

        总结：[总结]

        注意：  
        列点数量应根据验证后的信息灵活调整，不限于模板中的示例数量。  
        确保所有信息经过验证，来源可靠，并在输出中提供链接。
    """,
)

system_message = """
    根据用户的问题，通过网络搜索验证信息，并按照下方指定模板输出最终答案，包含相关信息来源链接。  

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

    信息来源：[相关链接]

    总结：[总结]

    注意：  
    列点数量应根据验证后的信息灵活调整，不限于模板中的示例数量。  
    确保所有信息经过验证，来源可靠，并在输出中提供链接。
"""

# 测试用例
# user_input = agent.invoke("What's the current time?")
# user_input = agent.invoke("当前美国总统是谁？")
# user_input = agent.invoke({"input": "Who is the current President of the United States?"})
# user_input = agent.invoke({"input": "当前美国总统是谁？"})
# user_input = agent.invoke({"input": "What's the current time?"})
# user_input = "请推荐一个食谱给我。"
user_input = "请为我详细介绍一下黑松露。"

input = {
    "system_msg": system_message,
    "user_input": user_input,
}

# response = agent.invoke({"input": input})
response = agent.invoke(prompt_template.format(user_input=user_input))
print("Final Answer:", response['output'])


在Web应用中实现流式响应
使用FastAPI的StreamingResponse将输出实时传输到客户端。
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import queue
import threading

app = FastAPI()

class QueueCallbackHandler(BaseCallbackHandler):
    """将回调数据存入队列供生成器读取"""
    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_agent_action(self, action, **kwargs):
        self.q.put(f"\nAction: {action.log}")

    def on_tool_end(self, output: str, **kwargs):
        self.q.put(f"\nObservation: {output}")

    def on_agent_finish(self, finish, **kwargs):
        self.q.put(f"\nFinal Result: {finish.return_values['output']}")
        self.q.put(None)  # 结束信号

@app.get("/chat")
def chat_stream(query: str):
    q = queue.Queue()
    handler = QueueCallbackHandler(q)

    # 初始化带流式回调的Agent
    llm = OpenAI(streaming=True, callbacks=[handler], temperature=0)
    tools = load_tools(["serpapi"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, callbacks=[handler])

    # 在后台线程中运行Agent
    def run_agent():
        agent.run(query)

    thread = threading.Thread(target=run_agent)
    thread.start()

    # 生成器函数，从队列中获取数据并yield
    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"
        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")
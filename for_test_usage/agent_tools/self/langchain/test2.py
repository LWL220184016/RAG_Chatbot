# 安装必要库
# pip install langchain-community duckduckgo-search tenacity

from langchain.agents import AgentType, initialize_agent, tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import Tool
from langchain_ollama import OllamaLLM
from tenacity import retry, stop_after_attempt, wait_fixed

from tools.duckduckgo_searching import duckduckgo_search

# 初始化模型（降低随机性）
llm = OllamaLLM(
    # model="llama3-groq-tool-use",
    # model="deepseek-r1_14b_FYP4",
    # model="llama3.2:1b",
    # model="deepseek-r1:32b",
    model="deepseek-r1:14b",
    temperature=0.0,
    top_p=0.9,
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
)

sys_msg = """系统消息：通过网络搜索验证信息，
    如果用户要求推荐食谱，你需要通过列点的方式详细说明食谱的每个步骤，食材原料，分量，时间，以及推荐的原因。
    如果用户要求的是食物的信息，你需要通过列点的方式详细说明食物的营养成分，热量，可能存在的问题等食物相关信息。
    ；
"""

# 测试用例
# response = agent.invoke("What's the current time?")
# response = agent.invoke("当前美国总统是谁？")
# response = agent.invoke({"input": "Who is the current President of the United States?"})
# response = agent.invoke({"input": "当前美国总统是谁？"})
# response = agent.invoke({"input": "What's the current time?"})
response = agent.invoke({"input": sys_msg + "用户输入：给我一个健康减肥的食谱。"})
print("Final Answer:", response['output'])
import os
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import Tool, DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3-groq-tool-use",
    # model="llama3.2:1b",
)

# 加载工具
def get_current_time():
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="get_current_time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
]
# tools = load_tools(["ddg-search", "llm-math"], llm=llm)

# 初始化 Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",  # 使用 zero-shot-react 类型的 Agent
    verbose=True,
    handle_parsing_errors=True,
)

# 运行 Agent
# response = agent.invoke("What is the height difference between Eiffel Tower and Taiwan 101 Tower?")
response = agent.run("What is the current time?")
print(response)
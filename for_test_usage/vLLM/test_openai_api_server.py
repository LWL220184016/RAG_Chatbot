import openai
import os
from typing import Iterator

# 設置 API 基礎 URL 指向本地 vLLM 服務器
# 通常本地 vLLM 服務默認在 8000 端口運行
openai.base_url = "http://localhost:8000/v1/"

# 可選: 設置一個虛擬的 API 密鑰，對於本地 vLLM 服務可能不是必須的
openai.api_key = "dummy-api-key"

# python -m vllm.entrypoints.openai.api_server     \
#         --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B     \
#         --port 8000     \
#         --gpu_memory_utilization 0.98     \
#         --quantization fp8     \
#         --max_model_len 80000

def generate_response_stream(prompt: str) -> Iterator[str]:
    """使用本地 vLLM 服務生成流式回應"""
    try:
        # 使用流式 API
        stream = openai.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # vLLM 服務中加載的模型名稱
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50000,
            stream=True  # 啟用流式輸出
        )
        
        # 處理流式響應
        collected_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                collected_content += content_piece
                yield content_piece
        
        return collected_content
    except Exception as e:
        yield f"錯誤: {str(e)}"

def generate_response(prompt: str) -> str:
    """使用本地 vLLM 服務生成完整回應 (非流式)"""
    collected_response = ""
    for chunk in generate_response_stream(prompt):
        collected_response += chunk
    return collected_response

# 測試流式輸出的函數
def test_stream(user_input):
    print(f"\n\033[38;5;208m😄問題: {user_input}\033[0m")  # 橙色高亮 (256-color)
    print(f"\n\033[38;5;46m🤖回答: \033[0m", end="", flush=True)  # 鮮綠色
    for text_chunk in generate_response_stream(user_input):
        print(f"\033[38;5;46m{text_chunk}\033[0m", end="", flush=True)  # 去掉換行符號
    print()  # 最後添加一個換行

# 測試函數
if __name__ == "__main__":
    # 選擇測試方式：流式或完整響應
    use_streaming = True
    
    user_input = "你好！"
    if use_streaming:
        test_stream(user_input)
    else:
        response = generate_response(user_input)
        print(f"問題: {user_input}")
        print(f"回答: {response}")
    

    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        if use_streaming:
            test_stream(user_input)
        else:
            response = generate_response(user_input)
            print(f"問題: {user_input}")
            print(f"回答: {response}")
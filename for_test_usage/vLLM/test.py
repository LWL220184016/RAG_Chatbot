from vllm import LLM, SamplingParams

llm = LLM(
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    quantization="fp8",  # 可选: "awq" 或 "gptq"
    dtype="float16",  # 或者使用 "bfloat16"
    gpu_memory_utilization=0.98,  # 可选: 控制 GPU 内存使用率
    max_model_len=20480, 
)

# 可选: 定义采样参数
sampling_params = SamplingParams( 
    temperature=0.7, 
    top_p=0.95, 
    max_tokens=1024, 
) 

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    else:
        output = llm.generate(user_input, sampling_params=sampling_params)
        response = output[0].outputs[0].text
        print(f"\n\033[38;5;208m🔍 LLM: {response}\033[0m")  # 橙色高亮 (256-color)
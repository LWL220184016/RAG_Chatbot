import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype = torch.float16, 
    device_map = "cuda:0", 
    load_in_8bit = True
) # 移除 load_in_4bit
streamer = TextStreamer(tokenizer)



# input_text = "人、狼、羊、白菜要从河的此岸借由一艘船渡河至另一岸，其中只有人会划船，每次人只能带一件东西搭船渡河， 且狼和羊、羊和白菜不能在无人监视的情况下放在一起。 在这些条件下，在最小渡河次数下如何才能让大家都渡河至另一河岸?"  # 或者 "Hello"
# input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda:0")
# attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to("cuda:0") # 确保 pad_token_id 已设置

# output = model.generate(
#     input_ids, 
#     attention_mask=attention_mask, 
#     max_length=5000, 
#     pad_token_id=tokenizer.pad_token_id, 
#     streamer=streamer, 
# ) # 显式传入 pad_token_id
# output_text = tokenizer.decode(output, skip_special_tokens=True)
# print("Simplified Output:", output_text)



input_text = "旁邊的字串中有多少英文字母s？ 'c489sAWE30jkxxs294#(jnfk)39s989))'"  # 或者 "Hello"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda:0")
attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to("cuda:0") # 确保 pad_token_id 已设置

output = model.generate(
    input_ids, 
    attention_mask=attention_mask, 
    max_length=5000, 
    pad_token_id=tokenizer.pad_token_id, 
    streamer=streamer, 
) # 显式传入 pad_token_id
output_text = tokenizer.decode(output, skip_special_tokens=True)
print("Simplified Output:", output_text)
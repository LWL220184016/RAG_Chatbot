from airllm import AutoModel

MAX_LENGTH = 64
# could use hugging face model repo id:
# model = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", compression='4bit')

# or use model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

# input_text = [
#         'What is the capital of United States?',
#         #'I like',
#     ]

input_text = ["有一個農夫帶著一隻狼、一隻羊和一顆白菜要過河，但他每次只能帶其中一樣東西過河。如果他把狼和羊留在同一邊，狼會吃掉羊；如果他把羊和白菜留在同一邊，羊會吃掉白菜。農夫該如何安全地將所有東西都帶到對岸？ " \
            "請把問題分開一步步解決，想清楚每一步會造成什麽結果，農夫過一次河為一個步驟，一個來回是兩個步驟"]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
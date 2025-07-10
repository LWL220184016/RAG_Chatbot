"""
A minimal RLHF (Reinforcement Learning from Human Feedback) training script for language models using Hugging Face Transformers and trl (PPOTrainer).
This script demonstrates self-chat with tool use, time sequence, and natural topic ending: the model generates a topic, can search, discuss, or naturally end the conversation, and scores the output.
Install requirements: pip install transformers datasets trl torch requests beautifulsoup4
"""
import torch
import requests
import re
import multiprocessing

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig

# Configurations
MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer

code in ----- may able import from chatbot_congig.py
-------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

LLM1_output_queue = multiprocessing.Queue()
LLM2_output_queue = multiprocessing.Queue()

should able to use llm_process_func_ws
LLM1_process = multiprocessing.Process(target=, args={LLM1_output_queue, LLM2_output_queue})
LLM2_process = multiprocessing.Process()

System_message = "; "
-------------------------------------------------------------------------------------------


def dummy_reward_fn(history, action, ended):
    raise NotImplementedError("Implement your own reward function based on action and history.")

def duckduckgo_search(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = soup.find_all('a', class_='result__a')
    snippets = [a.get_text() for a in results[:3]]
    return "\n".join(snippets) if snippets else "No relevant results found."

# PPO Config
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
)

# PPO Trainer
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    config=ppo_config,
)

# Self-chat with tool use, time sequence, and natural ending
for epoch in range(2):
    history = []
    topic_prompt = "請提出一個值得與熟悉的人討論的時事或科學主題。"
    topic_ids = model.generate(tokenizer(topic_prompt, return_tensors="pt").to(DEVICE), max_new_tokens=16, do_sample=True)
    topic = tokenizer.decode(topic_ids[0], skip_special_tokens=True).replace(topic_prompt, "").strip()
    print(f"[Model Topic] {topic}")
    history.append(f"[Topic] {topic}")

    for turn in range(4):
        context = "\n".join(history[-3:])  # last 3 turns
        action_prompt = f"主題：{topic}\n對話歷史：{context}\n請像和熟悉的朋友討論一樣，根據需要可以查詢網路、討論、或自然結束話題。"
        action_ids = model.generate(tokenizer(action_prompt, return_tensors="pt").to(DEVICE), max_new_tokens=64, do_sample=True)
        action = tokenizer.decode(action_ids[0], skip_special_tokens=True).replace(action_prompt, "").strip()
        print(f"[Model Action] {action}")

        ended = is_natural_ending(action)
        used_search = "查詢" in action or "搜尋" in action or "search" in action.lower()
        web_snippet = ""
        if used_search:
            # Extract search query from action or generate one
            search_query_prompt = f"根據主題和對話歷史產生一個網路搜尋查詢：主題：{topic}\n對話歷史：{context}\n查詢："
            search_ids = model.generate(tokenizer(search_query_prompt, return_tensors="pt").to(DEVICE), max_new_tokens=12, do_sample=True)
            search_query = tokenizer.decode(search_ids[0], skip_special_tokens=True).replace(search_query_prompt, "").strip()
            print(f"[Search Query] {search_query}")
            web_snippet = duckduckgo_search(search_query)
            print(f"[Web Search Result]\n{web_snippet}")
            action += f"\n[Web] {web_snippet}"

        # Step: Score the action
        reward = dummy_reward_fn(history, action, ended)
        print(f"Reward: {reward}\n---")

        # PPO step
        ppo_trainer.step([action_prompt], [action], [reward])
        history.append(action)
        if ended:
            break

print("RLHF self-chat with tool use, time sequence, and natural ending training demo complete.")

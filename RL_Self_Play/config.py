# RL_Self_Play/config.py

# 包含了所有模型、訓練和環境的設定
import torch

# ==================================
# 基本模型設定
# ==================================
# 我們將使用一個較小的模型來進行演示，您可以換成任何支援的 Hugging Face 模型
# 例如: "meta-llama/Llama-2-7b-chat-hf" 或您自己的模型
MODEL_ID = "gpt2" 
TOKENIZER_ID = "gpt2"

# ==================================
# PPO 訓練設定 (TRL)
# ==================================
PPO_CONFIG = {
    "batch_size": 1,
    "forward_batch_size": 1,
    "ppo_epochs": 4,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "kl_penalty": "kl",
    "log_with": None,  # 可以設定為 "wandb"
    "use_score_scaling": True,
    "use_score_norm": True,
}

# ==================================
# 對話生成設定
# ==================================
GENERATION_KWARGS = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 50256,  # GPT-2 的 pad_token_id
    "max_new_tokens": 64,
}

# ==================================
# 自我對弈環境設定
# ==================================
# 初始對話提示
INITIAL_PROMPT = "你好，我們來聊聊關於學習的話題吧。"

# 最大對話輪次
MAX_TURNS = 5

# 訓練設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# 訓練流程設定
# ==================================
# 總共進行多少次自我對弈的完整對話
TOTAL_EPISODES = 100

# 每隔多少次對話後儲存一次模型
SAVE_FREQ = 10

# 模型儲存路徑
OUTPUT_DIR = "ppo_self_play_model"

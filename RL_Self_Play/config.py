# RL_Self_Play/config.py

# 包含了所有模型、訓練和環境的設定
import torch

# ==================================
# 基本模型設定
# ==================================
# 我們將使用一個較小的模型來進行演示，您可以換成任何支援的 Hugging Face 模型
# 例如: "meta-llama/Llama-2-7b-chat-hf" 或您自己的模型
MODEL_ID = "Qwen/Qwen3-1.7B" 
TOKENIZER_ID = "Qwen/Qwen3-1.7B"

# ==================================
# PPO 訓練設定 (TRL)
# ==================================
PPO_CONFIG = {
    "batch_size": 1,
    "num_ppo_epochs": 4,
    "learning_rate": 1.41e-5,
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

# 最大對話輪次
MAX_TURNS = 100

# 訓練設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# 訓練流程設定
# ==================================
# 總共進行多少次自我對弈的完整對話
TOTAL_EPISODES = 20

# 每隔多少次對話後儲存一次模型
SAVE_FREQ = 10

# 模型儲存路徑
OUTPUT_DIR = "ppo_self_play_model"

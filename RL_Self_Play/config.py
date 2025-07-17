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
REWARD_MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
VALUE_MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"

# ==================================
# PPO 訓練設定 (TRL)
# ==================================
PPO_CONFIG = {
    "learning_rate": 1.41e-5,
    "batch_size": 1,
    "mini_batch_size": 1,
    "ppo_epochs": 4,
    "gradient_accumulation_steps": 1,
    "optimize_cuda_cache": True,
}

# ==================================
# 對話生成設定
# ==================================
GENERATION_KWARGS = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.7,
    "max_new_tokens": 64,
    "pad_token_id": None,  # 將在運行時設定
}

# ==================================
# 自我對弈環境設定
# ==================================

# 最大對話輪次
MAX_TURNS = 10

# 訓練設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# 訓練流程設定
# ==================================
# 總共進行多少次自我對弈的完整對話
TOTAL_EPISODES = 5

# 每隔多少次對話後儲存一次模型
SAVE_FREQ = 2

# 模型儲存路徑
OUTPUT_DIR = "ppo_self_play_model"

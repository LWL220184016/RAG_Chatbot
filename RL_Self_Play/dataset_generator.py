# RL_Self_Play/train.py
import os
import time # 引入 time 模組
import torch
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 從本地文件導入配置和類
from config import (
    MODEL_ID,
    TOKENIZER_ID,
    GENERATION_KWARGS,
    MAX_TURNS,
    DEVICE,
    TOTAL_EPISODES,
)
from agent import DialogueAgent
from reward_model import RewardModel
from final.Data_Storage.json_memory import JSON_Memory
from final.LLM.prompt_template import Message

def main():
    """
    主訓練函數，協調整個自我對弈和強化學習微調流程。
    """
    # 1. 初始化模型、Tokenizer 和 PPO 訓練器
    # =================================================================
    
    # 加載預訓練模型和 tokenizer
    # AutoModelForCausalLMWithValueHead 會在基礎語言模型上添加一個價值頭 (value head)
    # 這個價值頭用於在 PPO 訓練中估計狀態的價值
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    # 設定 pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    agent = DialogueAgent(model, tokenizer, DEVICE)

    # 這裡使用 User 是因為大多數模型訓練時使用的是 User, Assistant 和 System, 無法辨識訓練時沒出現的 Role
    # 下方使用 agent1 和 agent2 是因為方便程式碼編寫, 提高程式效率並且已經在提示詞中告訴模型他的角色
    msg = Message("User")
    temp_memory_handler = JSON_Memory("temp_memory")
    chat_history_recorder = JSON_Memory("chat_history_record")
    
    # 初始化上次說話時間和說話者
    last_speech_time = time.time()
    content = " "

    for i in range(TOTAL_EPISODES):
        
        # 每個 episode 包含多輪對話
        for turn in range(MAX_TURNS):
            agent_id = turn % 2  # 0 或 1，表示兩個分身
            print(f"\n--- Turn {turn+1} ---")

            time_since_last_speech = time.time() - last_speech_time
            
            prompt, timestamp = msg.update_content(content=content, memory=temp_memory_handler.get())
            # 將時間資訊加入到提示詞中，讓模型學習計時
            prompt[1]["content"] = (
                f"You are agent{agent_id}. "
                f"It has been {time_since_last_speech:.1f} seconds since the last turn. "
                + prompt[1]["content"]
            )
            
            print(f"Debug: 分身 {agent_id} 的提示: {prompt}")
            response_text, query_tensor, response_tensor = agent.generate_response(
                prompt, GENERATION_KWARGS
            )
            
            # 記錄生成的回應，無論內容如何
            # 如果回應為空或僅有空白，視為 "無輸出"
            is_silent = not response_text.strip()
            if is_silent:
                response_text = "[沉默]"

            print(f"分身 {agent_id} (潛在回應): {response_text}")
            time.sleep(0.1) # 模擬思考延遲

            # Role 是 "agent(agent_id)" 的原因, 請翻看程式碼 msg = Message("User") 處的註解
            temp_memory_handler.add(message=response_text, Role=f"agent{str(agent_id)}", timestamp=timestamp)
            chat_history_recorder.add_no_limit(message=response_text, Role=f"agent{str(agent_id)}", timestamp=timestamp)

    print("訓練完成！")

if __name__ == "__main__":
    main()

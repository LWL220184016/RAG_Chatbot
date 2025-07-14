# RL_Self_Play/train.py
import os
import time # 引入 time 模組
import torch
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

# 從本地文件導入配置和類
from config import (
    MODEL_ID,
    TOKENIZER_ID,
    PPO_CONFIG,
    GENERATION_KWARGS,
    MAX_TURNS,
    DEVICE,
    TOTAL_EPISODES,
    SAVE_FREQ,
    OUTPUT_DIR,
)
from agent import DialogueAgent
from reward_model import SimpleRewardModel
from final.Data_Storage.json_memory import JSON_Memory
from final.LLM.prompt_template import Message

def main():
    """
    主訓練函數，協調整個自我對弈和強化學習微調流程。
    """
    # 1. 初始化模型、Tokenizer 和 PPO 訓練器
    # =================================================================
    ppo_config = PPOConfig(**PPO_CONFIG)
    
    # 加載預訓練模型和 tokenizer
    # AutoModelForCausalLMWithValueHead 會在基礎語言模型上添加一個價值頭 (value head)
    # 這個價值頭用於在 PPO 訓練中估計狀態的價值
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    # 設定 pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer)
    agent = DialogueAgent(model, tokenizer, DEVICE)
    reward_model = SimpleRewardModel(device=DEVICE)

    # 這裡使用 User 是因為大多數模型訓練時使用的是 User, Assistant 和 System, 無法辨識訓練時沒出現的 Role
    # 下方使用 agent1 和 agent2 是因為方便程式碼編寫, 提高程式效率並且已經在提示詞中告訴模型他的角色
    msg = Message("User")
    temp_memory_handler = JSON_Memory("temp_memory")
    chat_history_recorder = JSON_Memory("chat_history_record")
    
    # 初始化上次說話時間和說話者
    last_speech_time = time.time()
    last_speaker_id = None
    content = " "

    for episode in tqdm(range(TOTAL_EPISODES), desc="自我對弈訓練"):
        
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

            # =================================================================
            # 決策與更新：
            # 我們不再隨機選擇，而是將兩個代理的生成結果都納入學習過程，
            # 並都添加到對話日誌中，以體現時間的流動。
            # =================================================================

            if is_silent:
                # time_since_last_speech 已移至前方計算
                # 如果是另一個代理剛說完話，那麼當前代理保持沉默是合理的
                if last_speaker_id is not None and last_speaker_id != agent_id:
                    reward_value = 0.1 # 沉默以避免打斷對方
                # 避免連續說話（同一代理）
                elif time_since_last_speech < 1: # 小於1秒
                    reward_value = -0.5
                # 避免太久不說話
                elif time_since_last_speech > 5: # 大於5秒
                    reward_value = -0.2
                else:
                    reward_value = 0.1 # 在理想時間範圍內不說話
            else:
                # 有說話，根據內容給予獎勵，並更新說話時間和說話者
                reward_value = reward_model.get_reward(response_text)
                last_speech_time = time.time()
                last_speaker_id = agent_id

            reward = torch.tensor([reward_value], dtype=torch.float, device=DEVICE)

            # 使用 PPO 進行單步訓練
            train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])

            # Role 是 "agent(agent_id)" 的原因, 請翻看程式碼 msg = Message("User") 處的註解
            temp_memory_handler.add(message=response_text, Role=f"agent{str(agent_id)}", timestamp=timestamp)
            chat_history_recorder.add_no_limit(message=response_text, Role=f"agent{str(agent_id)}", timestamp=timestamp)

        # 每隔一段時間儲存模型
        if (episode + 1) % SAVE_FREQ == 0:
            output_dir = os.path.join(OUTPUT_DIR, f"episode_{episode+1}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"模型已儲存至 {output_dir}")

    print("訓練完成！")

if __name__ == "__main__":
    main()

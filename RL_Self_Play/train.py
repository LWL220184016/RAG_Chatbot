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
    REWARD_MODEL_ID, 
    PPO_CONFIG,
    GENERATION_KWARGS,
    MAX_TURNS,
    DEVICE,
    TOTAL_EPISODES,
    SAVE_FREQ,
    OUTPUT_DIR,
)
from agent import DialogueAgent
from reward_model import RewardModel
from final.Data_Storage.json_memory import JSON_Memory

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
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID)
    if torch.cuda.is_available():
        model = model.to(DEVICE)
    
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID)
    if torch.cuda.is_available():
        model_ref = model_ref.to(DEVICE)
    
    reward_model = RewardModel(model_name=REWARD_MODEL_ID, device=DEVICE)
    
    # 設定參考模型為不可訓練
    for param in model_ref.parameters():
        param.requires_grad_(False)
        
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    # 設定 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 更新生成參數中的 pad_token_id
    generation_kwargs = GENERATION_KWARGS.copy()
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    # 創建 PPO 訓練器
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=model_ref,
        tokenizer=tokenizer,
    )
    agent = DialogueAgent(model, tokenizer, DEVICE)

    # 初始化對話系統
    temp_memory_handler = JSON_Memory("temp_memory")
    chat_history_recorder = JSON_Memory("chat_history_record")
    
    # 初始化上次說話時間和說話者
    last_speech_time = time.time()
    last_speaker_id = None
    
    # 初始對話主題列表
    conversation_topics = [
        "今天天氣如何？",
        "你最喜歡的食物是什麼？", 
        "最近有什麼有趣的事情嗎？",
        "你對人工智能有什麼看法？",
        "你平時喜歡做什麼？"
    ]

    for episode in tqdm(range(TOTAL_EPISODES), desc="自我對弈訓練"):
        print(f"\n=== Episode {episode + 1} ===")
        
        # 每個episode開始時清空臨時記憶，開始新對話
        temp_memory_handler.clear()
        
        # 選擇對話主題
        topic = conversation_topics[episode % len(conversation_topics)]
        print(f"對話主題: {topic}")
        
        # 初始化對話歷史
        conversation_history = [
            {"role": "system", "content": "你是一個友善的對話機器人，會進行自然的對話。"},
            {"role": "user", "content": topic}
        ]
        
        # 每個 episode 包含多輪對話
        for turn in range(MAX_TURNS):
            agent_id = turn % 2  # 0 或 1，表示兩個分身
            print(f"\n--- Turn {turn+1}, Agent {agent_id} ---")

            time_since_last_speech = time.time() - last_speech_time
            
            # 構建當前agent的提示
            current_conversation = conversation_history.copy()
            
            # 添加角色特定的系統提示
            agent_prompt = f"你是Agent {agent_id}，正在與另一個AI進行對話。已經過去了 {time_since_last_speech:.1f} 秒。請自然地回應。"
            current_conversation.append({"role": "system", "content": agent_prompt})
            
            print(f"Debug: Agent {agent_id} 的對話歷史長度: {len(current_conversation)}")
            
            response_text, query_tensor, response_tensor = agent.generate_response(
                current_conversation, generation_kwargs
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

            reward = torch.tensor([reward_value], dtype=torch.float)
            if torch.cuda.is_available():
                reward = reward.to(DEVICE)

            # 使用 PPO 進行單步訓練
            stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])

            # 記錄對話歷史
            timestamp = datetime.datetime.now().isoformat()
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # 保存到記憶系統
            temp_memory_handler.add(message=response_text, Role=f"agent{str(agent_id)}", timestamp=timestamp)
            chat_history_recorder.add_no_limit(message=response_text, Role=f"agent{str(agent_id)}", timestamp=timestamp)
            
            print(f"Agent {agent_id} 回應: {response_text}")
            print(f"獎勵值: {reward_value:.3f}")
            
            # 如果對話變得太長，截斷歷史
            if len(conversation_history) > 20:
                # 保留系統消息和最近的10條消息
                conversation_history = conversation_history[:2] + conversation_history[-10:]

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

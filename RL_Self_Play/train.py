# RL_Self_Play/train.py
import os
import time # 引入 time 模組
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

# 從本地文件導入配置和類
from config import (
    MODEL_ID,
    TOKENIZER_ID,
    PPO_CONFIG,
    GENERATION_KWARGS,
    INITIAL_PROMPT,
    MAX_TURNS,
    DEVICE,
    TOTAL_EPISODES,
    SAVE_FREQ,
    OUTPUT_DIR,
)
from agent import DialogueAgent
from reward_model import SimpleRewardModel

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
    
    # 初始化 PPO 訓練器
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer)

    # 2. 初始化兩個對話代理 (分身)
    # =================================================================
    # 兩個代理共享同一個模型和 tokenizer
    # 當模型被 PPO 訓練器更新時，兩個代理的能力會同時提升
    agent1 = DialogueAgent(model, tokenizer, DEVICE)
    agent2 = DialogueAgent(model, tokenizer, DEVICE)
    agents = [agent1, agent2]

    # 3. 初始化獎勵模型
    # =================================================================
    reward_model = SimpleRewardModel(device=DEVICE)

    # 4. 開始自我對弈訓練循環
    # =================================================================
    for episode in tqdm(range(TOTAL_EPISODES), desc="自我對弈訓練"):
        
        # 初始化對話歷史記錄，現在是一個列表
        dialogue_log = [{'agent_id': 'System', 'text': INITIAL_PROMPT, 'timestamp': time.time()}]
        
        # 每個 episode 包含多輪對話
        for turn in range(MAX_TURNS):
            
            # 從日誌構建當前的對話歷史字串，供模型輸入
            current_history_str = "\n".join([f"分身 {item['agent_id']}: {item['text']}" if item['agent_id'] != 'System' else item['text'] for item in dialogue_log])
            
            print(f"\n--- Turn {turn+1} ---")
            print(f"--- 當前對話歷史 ---\n{current_history_str}\n" + "="*30)

            # =================================================================
            # 並行運作模擬：
            # 讓兩個代理都對當前的歷史生成回應。
            # 核心思想是兩者都獨立思考，並記錄下所有思考結果。
            # =================================================================
            
            all_responses = []

            # 讓兩個 agent 都生成一個回應
            for i, agent in enumerate(agents):
                response_text, query_tensor, response_tensor = agent.generate_response(
                    current_history_str, GENERATION_KWARGS
                )
                
                # 記錄生成的回應，無論內容如何
                # 如果回應為空或僅有空白，視為 "無輸出"
                is_silent = not response_text.strip()
                if is_silent:
                    response_text = "[無輸出]"

                all_responses.append({
                    'agent_id': i + 1,
                    'text': response_text,
                    'timestamp': time.time(),
                    'query_tensor': query_tensor,
                    'response_tensor': response_tensor,
                    'is_silent': is_silent
                })
                
                print(f"分身 {i+1} (潛在回應): {response_text}")
                time.sleep(0.1) # 模擬思考延遲

            # =================================================================
            # 決策與更新：
            # 我們不再隨機選擇，而是將兩個代理的生成結果都納入學習過程，
            # 並都添加到對話日誌中，以體現時間的流動。
            # =================================================================

            # 訓練兩個代理的生成結果，並將它們都加入日誌
            for resp_data in all_responses:
                # 計算獎勵
                # 對於無輸出的情況，可以給予一個小的負獎勵或零獎勵
                reward_value = -0.1 if resp_data['is_silent'] else reward_model.get_reward(resp_data['text'])
                reward = torch.tensor([reward_value], dtype=torch.float, device=DEVICE)

                # 使用 PPO 進行單步訓練
                train_stats = ppo_trainer.step([resp_data['query_tensor'][0]], [resp_data['response_tensor'][0]], [reward])
                
                # 將此回應添加到對話日誌中
                dialogue_log.append({
                    'agent_id': resp_data['agent_id'],
                    'text': resp_data['text'],
                    'timestamp': resp_data['timestamp']
                })

            # 模擬真實對話的時間間隔
            print("...等待下一個對話回合...")
            time.sleep(1)

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

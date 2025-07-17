#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試自我對弈強化學習系統
"""

import os
import sys
import torch

# 添加父目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

def test_imports():
    """測試所有必要的模組是否能正常導入"""
    print("測試模組導入...")
    
    try:
        from config import MODEL_ID, TOKENIZER_ID, DEVICE
        print(f"✓ 配置導入成功: MODEL_ID={MODEL_ID}")
        
        from agent import DialogueAgent
        print("✓ DialogueAgent 導入成功")
        
        from reward_model import RewardModel
        print("✓ RewardModel 導入成功")
        
        print(f"✓ 設備: {DEVICE}")
        return True
        
    except Exception as e:
        print(f"✗ 導入失敗: {e}")
        return False

def test_reward_model():
    """測試獎勵模型"""
    print("\n測試獎勵模型...")
    
    try:
        from reward_model import RewardModel
        
        reward_model = RewardModel()
        
        # 測試簡單獎勵
        reward1 = reward_model.get_reward("你好，很高興見到你！")
        print(f"✓ 簡單獎勵測試: {reward1}")
        
        # 測試沉默獎勵
        reward2 = reward_model.get_reward("[沉默]")
        print(f"✓ 沉默獎勵測試: {reward2}")
        
        return True
        
    except Exception as e:
        print(f"✗ 獎勵模型測試失敗: {e}")
        return False

def test_agent():
    """測試對話代理"""
    print("\n測試對話代理...")
    
    try:
        from agent import DialogueAgent
        from config import MODEL_ID, TOKENIZER_ID, DEVICE
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead
        
        # 載入模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID)
        if torch.cuda.is_available():
            model = model.to(DEVICE)
            
        agent = DialogueAgent(model, tokenizer, DEVICE)
        
        # 測試生成回應
        test_history = [
            {"role": "system", "content": "你是一個友善的AI助手。"},
            {"role": "user", "content": "你好！"}
        ]
        
        generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": 32,
            "temperature": 0.7,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        response, query_tensor, response_tensor = agent.generate_response(
            test_history, generation_kwargs
        )
        
        print(f"✓ 代理生成測試成功: {response}")
        return True
        
    except Exception as e:
        print(f"✗ 代理測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=== 自我對弈強化學習系統測試 ===\n")
    
    success_count = 0
    total_tests = 3
    
    if test_imports():
        success_count += 1
        
    if test_reward_model():
        success_count += 1
        
    if test_agent():
        success_count += 1
        
    print(f"\n=== 測試結果: {success_count}/{total_tests} 通過 ===")
    
    if success_count == total_tests:
        print("✓ 所有測試通過！系統已準備好進行訓練。")
        print("\n要開始訓練，請運行: python train.py")
    else:
        print("✗ 部分測試失敗，請檢查錯誤並修復。")

if __name__ == "__main__":
    main()

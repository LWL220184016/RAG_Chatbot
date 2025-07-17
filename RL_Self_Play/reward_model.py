import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModel:
    """
    一個簡單的獎勵模型範例。
    在真實場景中，您可能會使用一個訓練好的模型來評估回應的品質。
    這裡我們使用一個基於關鍵字的簡單規則來給予獎勵。
    """
    def __init__(self, model_name = None, device = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if model_name is None:
            # model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
            model_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_reward(self, response: str, prompt: str = None, sys_msg: str = None) -> float:
        """
        根據回應文本計算獎勵分數。

        Args:
            response (str): 模型生成的回應。
            prompt (str): 用戶輸入的提示（可選）。
            sys_msg (str): 系統消息（可選）。

        Returns:
            float: 獎勵分數。
        """
        
        # 如果沒有提供 prompt 和 sys_msg，使用簡單的規則給分
        if prompt is None or sys_msg is None:
            return self._simple_reward(response)
        
        conv = [{"role": "user", "content": prompt}, {"role": "system", "content": sys_msg}, {"role": "assistant", "content": response}]

        # Format and tokenize the conversations
        conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)

        print("conv_formatted: ", conv_formatted)

        # These two lines remove the potential duplicate bos token
        if self.tokenizer.bos_token is not None and conv_formatted.startswith(self.tokenizer.bos_token):
            conv_formatted = conv_formatted[len(self.tokenizer.bos_token):]
        conv_tokenized = self.tokenizer(conv_formatted, return_tensors="pt").to(self.device)

        # Get the reward scores
        with torch.no_grad():
            score = self.reward_model(**conv_tokenized).logits[0][0].item()
        print(f"Score for response : {score}")
        return score
    
    def _simple_reward(self, response: str) -> float:
        """
        簡單的基於規則的獎勵函數。
        """
        if not response or response == "[沉默]":
            return 0.0
        
        # 簡單的獎勵規則
        score = 0.5  # 基礎分數
        
        # 鼓勵有意義的回應
        if len(response.strip()) > 5:
            score += 0.2
        
        # 鼓勵禮貌用語
        polite_words = ["請", "謝謝", "對不起", "不好意思", "please", "thank", "sorry"]
        if any(word in response.lower() for word in polite_words):
            score += 0.3
        
        # 懲罰過於簡短或重複的回應
        if len(response.strip()) < 3:
            score -= 0.3
        
        return max(0.0, min(1.0, score))


if __name__ == '__main__':

    reward_model = RewardModel()

    sys_msg = "Returns a fraction indicating how talk like a long-time friend."

    prompt = "今天要一起跑嗎？"
    response1 = "好，出發啦！！！"
    response2 = "哈哈哈哈，你小心不要讓我把你落得太遠！"
    reward = reward_model.get_reward(response1, prompt, sys_msg)
    print(f"Reward for response1: {reward}")

    prompt = "Do you want to run together today?"
    response1 = "Ok, let's go!!!"
    response2 = "Hahahaha, you? Don't let me leave you too far behind."
    reward = reward_model.get_reward(response1, prompt, sys_msg)
    print(f"Reward for response2: {reward}")

    # 測試簡單獎勵功能
    simple_reward = reward_model.get_reward("謝謝你的幫助！")
    print(f"Simple reward: {simple_reward}")

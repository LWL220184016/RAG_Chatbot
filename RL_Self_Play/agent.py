# RL_Self_Play/agent.py
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

class DialogueAgent:
    """
    對話代理，代表一個參與對話的分身。
    它包含了生成回應所需的神經網路模型和 tokenizer。
    """
    def __init__(self, model, tokenizer, device):
        """
        初始化一個對話代理。

        Args:
            model (AutoModelForCausalLMWithValueHead): 用於生成回應的價值頭模型。
            tokenizer (AutoTokenizer): 用於文本編碼和解碼的 tokenizer。
            device (torch.device): 模型運行的設備 (CPU or GPU)。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, history, generation_kwargs: dict):
        """
        根據對話歷史生成回應。

        Args:
            history: 當前的對話歷史（消息列表）。
            generation_kwargs (dict): 傳遞給 model.generate 的參數。

        Returns:
            tuple:
                - response_text (str): 生成的回應文本。
                - query_tensor (torch.Tensor): 編碼後的輸入張量。
                - response_tensor (torch.Tensor): 編碼後的回應張量。
        """
        # 將歷史文本編碼為輸入 ID
        try:
            query_text = self.tokenizer.apply_chat_template(
                history, 
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            # 如果 apply_chat_template 失敗，使用簡單的文本拼接
            query_text = ""
            for msg in history:
                if isinstance(msg, dict) and "content" in msg:
                    query_text += f"{msg.get('role', 'user')}: {msg['content']}\n"
            query_text += "assistant: "
        
        print("DEBUG query_text: ", query_text)
        query_tensor = self.tokenizer.encode(query_text, return_tensors="pt")
        if torch.cuda.is_available():
            query_tensor = query_tensor.to(self.device)
        
        # 使用模型生成回應
        # 我們傳遞 query_tensor 作為輸入
        with torch.no_grad():
            response_tensor = self.model.generate(query_tensor, **generation_kwargs)
        
        # 解碼生成的回應，並去除輸入部分
        response_text = self.tokenizer.decode(response_tensor[0][query_tensor.shape[1]:], skip_special_tokens=True)
        
        return response_text, query_tensor, response_tensor

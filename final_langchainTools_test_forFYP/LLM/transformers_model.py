import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm import LLM
import threading

class TF_LLM(LLM):
    def __init__(
            self, 
            model_name,
            torch_dtype,
            device_map,
            trust_remote_code:bool = False,
            is_user_talking: threading.Event = None,
            stop_event: threading.Event = threading.Event(),
            speaking_event: threading.Event = threading.Event()
        ):
        super().__init__(model_name, is_user_talking, stop_event, speaking_event)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    @super
    def llm_output():
        pass
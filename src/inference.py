

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import extract_department

MODEL_NAME = "ybelkada/falcon-7b-sharded-bf16"


class ProductDepartmentPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def predict(self, product_name: str) -> str:
        prompt = f"{product_name} ->:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )

        outputs = self.model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=10,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        return extract_department(decoded)


if __name__ == "__main__":
    predictor = ProductDepartmentPredictor()
    print(predictor.predict("French Milled Oval Almond Gourmande Soap"))

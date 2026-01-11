
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from prepare_data import load_and_prepare_data
from utils import extract_department


MODEL_NAME = "ybelkada/falcon-7b-sharded-bf16"


def evaluate(sample_size=25):
    _, _, test_df = load_and_prepare_data()
    test_texts = list(test_df["text"])[:sample_size]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    outputs = pipeline(
        test_texts,
        max_length=100,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    predictions = [
        extract_department(out[0]["generated_text"])
        for out in outputs
    ]

    result_df = test_df.iloc[:sample_size][
        ["product_name", "department"]
    ].reset_index(drop=True)

    result_df["predicted_department"] = predictions
    print(result_df)


if __name__ == "__main__":
    evaluate()

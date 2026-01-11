import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from prepare_data import load_and_prepare_data

MODEL_NAME = "ybelkada/falcon-7b-sharded-bf16"
OUTPUT_DIR = "results"

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 512
MAX_STEPS = 120
LEARNING_RATE = 2e-4


def train():
    dataset, _, _ = load_and_prepare_data()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h"
        ]
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        fp16=True,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        report_to="none",
        group_by_length=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

    trainer.train()


if __name__ == "__main__":
    train()

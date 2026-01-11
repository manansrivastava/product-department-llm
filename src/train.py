

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig
from prepare_data import load_and_prepare_data


MODEL_NAME = "ybelkada/falcon-7b-sharded-bf16"
OUTPUT_DIR = "results"


def train():
    # Load dataset
    dataset, _, _ = load_and_prepare_data()


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
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
            "dense_4h_to_h",
        ],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        max_steps=120,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        group_by_length=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )

    # FP32 norm fix
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module.to(torch.float32)

    trainer.train()


if __name__ == "__main__":
    train()

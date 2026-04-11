# borrowed code from https://github.com/ZeroNLP/Eraser

import os
import json
import random
import transformers

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .datasets import TrainDataset, EvalDataset
from .collator import DataCollatorForSeq2Seq
from .trainer import LlamaTrainer


def eraser_unlearn(
    base_model,
    tokenizer,
    ref_device,
    harm_path,
    help_path,
    algn_path,
    output_dir="./outputs/eraser",
    epochs=5,
    lr=2e-5,
    harmful_threshold=1.8,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    batch_size=1,
    model_max_length=2048,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    train_device = next(base_model.parameters()).device

    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = TrainDataset(
        harm_path, help_path, algn_path, tokenizer, model_max_length
    )
    eval_dataset = EvalDataset(
        harm_path, help_path, algn_path, tokenizer, model_max_length
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        past_index=-1,
        dataloader_num_workers=0,
    )

    original_path = (
        base_model.config._name_or_path
        if hasattr(base_model.config, "_name_or_path")
        else base_model.config.name_or_path
    )

    trainer = LlamaTrainer(
        original_path=original_path,
        harmful_threshold=harmful_threshold,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        ref_device=ref_device,    
    )

    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"ERASER adapter saved to {output_dir}")

    return model.eval()
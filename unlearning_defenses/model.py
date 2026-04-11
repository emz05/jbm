# borrowed and refactored code from https://github.com/sail-sg/closer-look-LLM-unlearning

import csv
import copy
import torch
from .losses import get_loss
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from .trainer import CustomTrainerForgetting  
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from .dataset import JBBUnlearningDataset, jbb_collator

class CustomTrainerNoDS(CustomTrainerForgetting):
    def e_prepare_deepspeed(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _wrap_model(self, model, training=True, dataloader=None):
        return model


def train_unlearning(
    base_model_name,
    tokenizer,
    forget_data,       
    retain_data,
    loss_type,        
    forget_idk_data=None,   
    retain_idk_data=None,
    steps=100,
    tol_func=0.5, 
    tol_count=10,
    lr=1e-4,
    beta=0.1,
    forget_coeff=1.0,
    regularization_coeff=1.0,
    dtype=torch.bfloat16,
    device='cuda:0',
    ref_device='cuda:1',
):

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    )

    # Load train model
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    train_model = get_peft_model(base, lora_cfg)
    # train_model.print_trainable_parameters()

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": device},  
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    dataset = JBBUnlearningDataset(forget_data, retain_data, tokenizer,forget_idk_data=forget_idk_data,retain_idk_data=retain_idk_data,)
    batch_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=jbb_collator,
    )
    optimizer = AdamW(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=lr,
    )
    logs = []
    fp = f'./results/{loss_type}.csv'

    threshold_counter = 0
    train_model.train()
    step = 0
    print(f"{'Step':<10} {'Loss':<12} {'Forget':<12} {'Reg':<12}")
    
    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            forget_loss, regularization_loss = get_loss(
                train_model, ref_model, batch, loss_type, beta
            )
            if not torch.is_tensor(regularization_loss):
                regularization_loss = torch.tensor(
                    regularization_loss,
                    device=forget_loss.device,
                    dtype=forget_loss.dtype,
                )
            loss = (
                forget_coeff * forget_loss
                + regularization_coeff * regularization_loss
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss at step {step}: "
                    f"forget={forget_loss.item()} reg={regularization_loss.item()}"
                )

            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in train_model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            print(f"{step:<10} {loss.item():<12.6f} {forget_loss.item():<12.6f} {regularization_loss.item():<12.6f}")

            if forget_loss.item() < tol_func:
                threshold_counter += 1
                if threshold_counter >= tol_count:
                    print(
                        f"Terminating at step {step}, "
                        f"forget loss below {tol_func} "
                        f"for {tol_count} evals"
                    )
                    log = {
                        "step": step,
                        "loss": loss.item(),
                        "forget_loss": forget_loss.item(),
                        "reg_loss": regularization_loss.item()
                    }
                    logs.append(log)
                    return train_model.eval()
            else:
                threshold_counter = 0
    if logs:
        keys = logs[0].keys()
        with open(fp, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(logs)

    return train_model.eval()

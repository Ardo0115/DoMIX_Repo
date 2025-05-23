# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
import random
import numpy as np
from transformers import BitsAndBytesConfig

from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import math



"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

from peft import (  # noqa: E402
    LoraConfig,
    DoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402

from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import math




def set_seed(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy (if you're using it)
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If you're using CUDA, set the seed for GPU as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Make the CUDNN backend deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # Dora hyperparams
        dora_simple: bool = True,
        Wdecompose_target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        other_loras: List[str] = None, # paths to other loras to load
        mergeratio_init: float = 0.0, # initial value for mergeratios
        seed: int = 42, # for reproducibility
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"Wdecompose_target_modules: {Wdecompose_target_modules}\n"
        f"dora_simple: {dora_simple}"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"other_loras: {other_loras}\n"
        f"mergeratio_init: {mergeratio_init}\n"
        f"seed: {seed}\n"
    )
    # Set seed for  reproducibility
    set_seed(seed)

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=wandb_run_name,
            config={
                "base_model": base_model,
                "data_path": data_path,
                "output_dir": output_dir,
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "cutoff_len": cutoff_len,
                "val_set_size": val_set_size,
                "use_gradient_checkpointing": use_gradient_checkpointing,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": lora_target_modules,
                "Wdecompose_target_modules": Wdecompose_target_modules,
                "dora_simple": dora_simple,
                "bottleneck_size": bottleneck_size,
                "non_linearity": non_linearity,
                "adapter_dropout": adapter_dropout,
                "use_parallel_adapter": use_parallel_adapter,
                "use_adapterp": use_adapterp,
                "train_on_inputs": train_on_inputs,
                "scaling": scaling,
                "adapter_name": adapter_name,
                "target_modules": target_modules,
                "group_by_length": group_by_length,
            },
        )
        wandb.run.name = wandb_run_name


    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    
    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        # need to handle llama 3 separately
        if "Llama-3" in base_model:
            print("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    print(model)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "lora-merging":
        print("LoRA Merging init")
        assert other_loras is not None, "Please provide paths to other LoRAs"
        num_other_loras = len(other_loras)
        assert num_other_loras > 0, "Please provide paths to other LoRAs"
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            num_other_loras=num_other_loras,
            mergeratio_init=mergeratio_init
        )
    elif adapter_name == "dora":
        print("DoRA init")
        config = DoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, config)
    if adapter_name == "prefix-tuning":
        model.to('cuda')
    
    



    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    

    if adapter_name == "lora-merging":
        # Load other loras
        for i, lora_path in enumerate(other_loras):
            weights_to_load = torch.load(f'{lora_path}/adapter_model.bin')
            # Update keys to match the current model
            for k in list(weights_to_load.keys()):
                if 'lora_A' in k:
                    new_key = k.replace('lora_A.weight', f'lora_As_stacked')
                    weights_to_load[new_key] = weights_to_load.pop(k)
                elif 'lora_B' in k:
                    new_key = k.replace('lora_B.weight', f'lora_Bs_stacked')
                    weights_to_load[new_key] = weights_to_load.pop(k)
                else:
                    pass
            
            # Check if all keys are in the model
            for k in weights_to_load.keys():
                if k not in model.state_dict().keys():
                    print(f"Key {k} not in model")
                    raise KeyError(f"Key {k} not in model")
                
            # Copy the weights to the model
            with torch.no_grad():
                for k, v in weights_to_load.items():
                    for n, p in model.named_parameters():
                        if k in n:
                            p[i].copy_(v)
                            break
            
            # Check if all weights were copied
            for k in weights_to_load.keys():
                if not torch.all(torch.eq(weights_to_load[k], model.state_dict()[k][i])):
                    print(f"Weight {k} not copied correctly")
                    raise ValueError(f"Weight {k} not copied correctly")
            # Double by checking if the lora_stacked are not zero
            for n, p in model.named_parameters():
                if 'lora_As_stacked' in n or 'lora_Bs_stacked' in n:
                    if torch.all(torch.eq(p[i], torch.zeros_like(p[i]))):
                        print(f"Weight {n} not copied correctly")
                        raise ValueError(f"Weight {n} not copied correctly")
            
            print(f"Loaded LoRA {i} from {lora_path}")
        
        # Set mergeratios to be trainable
        for n, p in model.named_parameters():
            if 'mergeratios' in n:
                p.requires_grad = True

        if 'IdMixer' in output_dir:
            # Set the mixer to be the identity
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if 'lora_mergeMixer' in n:
                        p.copy_(torch.eye(p.shape[0], p.shape[1]))
                        print(f"Setting {n} to identity")

        if 'FreezeMixer' in output_dir:
            # Freeze the mixer
            for n, p in model.named_parameters():
                if 'lora_mergeMixer' in n:
                    p.requires_grad = False
                    print(f"Freezing {n}")

        if 'TrainAsCat' in output_dir:
            # Train the mixer as a cat
            for n, p in model.named_parameters():
                if 'lora_As_cat' in n:
                    p.requires_grad = True
                    print(f"Training {n}")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    set_seed(seed)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=train_val["train"].column_names)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=train_val["train"].column_names)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=train_val["train"].column_names)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    # Get 1/10 of the data from train_data
    # train_data = train_data.select(range(len(train_data)//3))
    # print(f"Training on {len(train_data)} samples")
    # warmup_steps = 6 if 'math' in data_path else 100
    # warmup_steps = 10
    set_seed(seed)

    def train_with_loop(model, tokenizer, train_data, val_data=None, resume_from_checkpoint=None):
        # === 기본 설정 ===
        model.train()
        model.config.use_cache = False
        model.to("cuda")
        
        # DataLoader
        train_dataloader = DataLoader(
            train_data,
            batch_size=micro_batch_size,
            shuffle=True,
            collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler
        num_training_steps = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        # === 학습 루프 ===
        global_step = 0
        completed_steps = 0
        model.zero_grad()

        for epoch in range(num_epochs):
            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(loop):
                batch = {k: v.to("cuda") for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    completed_steps += 1
                    global_step += 1

                    # Logging
                    loop.set_postfix(loss=loss.item() * gradient_accumulation_steps, step=global_step)
                    if use_wandb:
                        wandb.log({"train/loss": loss.item() * gradient_accumulation_steps, "train/learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)

                    # Evaluation (optional)
                    if val_data and global_step % eval_step == 0:
                        model.eval()
                        eval_loss = 0.0
                        val_loader = DataLoader(
                            val_data,
                            batch_size=micro_batch_size,
                            collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
                        )
                        for batch in val_loader:
                            batch = {k: v.to("cuda") for k, v in batch.items()}
                            with torch.no_grad():
                                with torch.cuda.amp.autocast(enabled=True):
                                    outputs = model(**batch)
                                    eval_loss += outputs.loss.item()
                        eval_loss /= len(val_loader)
                        print(f"[Eval @ step {global_step}] loss: {eval_loss}")
                        if use_wandb:
                            wandb.log({"eval/loss": eval_loss}, step=global_step)
                        model.train()

                    # Save checkpoint
                    if global_step % save_step == 0:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)

        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Training complete!")
    

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    
    train_with_loop(model, tokenizer, train_data, val_data)
    if use_wandb:
        wandb.finish()



def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
from approaches.finetune import Appr
from config import parseing_finetune
from utils import utils
import logging
import os
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, set_seed, AdamW
from dataloader.data import get_dataset, get_dataset_eval, dataset_class_num
import random
import numpy as np
import copy
from torch.utils.data import Sampler
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)
# Set up logger
logger = logging.getLogger(__name__)

import wandb
class CustomSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Set the seed for the current epoch
        seed = self.seed + self.epoch
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Generate shuffled indices
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch
def main():
    args = parseing_finetune()
    args = utils.model.prepare_sequence_finetune(args)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    if args.wandb:
        if args.hyperparameter_tune:
            wandb.init(project='Continual DAP Finetune sweep',
                config=args)
        else:
            wandb.init(project=args.project_name,
                config=args)
        if not args.hyperparameter_tune:
            wandb.run.name = "%s_%s_%s_%d" % (
                args.baseline, args.finetune_type, args.dataset_name, args.seed)
            wandb.run.save()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.log_dir is not None:
        handler = logging.FileHandler(args.log_dir)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # if use multilabel-classification datasets:
    if 'multi' in args.dataset_name:
        args.problem_type = 'multi_label_classification'

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
    accelerator.wait_for_everyone()

    # Get the datasets and process the data.
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer

    max_length = args.max_seq_length

    logger.info('==> Preparing data..')

    if args.hyperparameter_tune:
        datasets = get_dataset_eval(args.dataset_name, tokenizer=tokenizer, args=args)
    else:
        datasets = get_dataset(args.dataset_name, tokenizer=tokenizer, args=args)
    print(f'Dataset: {args.dataset_name}')

    print(f'Size of training set: {len(datasets["train"])}')
    print(f'Size of testing set: {len(datasets["test"])}')

    train_dataset = datasets['train']
    test_dataset = datasets['test']

    test_dataset = test_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    test_dataset.set_format(type='torch', columns=[
                            'input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=8)

    train_dataset = train_dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    train_dataset.set_format(type='torch', columns=[
                             'input_ids', 'attention_mask', 'labels'])
    sampler = CustomSampler(train_dataset, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False,
                              num_workers=8, sampler=sampler)  # consider batch size

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    if args.class_num is None:
        args.class_num = dataset_class_num[args.dataset_name]

    # Declare the model and set the training parameters.
    logger.info('==> Building model..')

    # Print random seed state before function call
    print("Random seed state before model initialization:", torch.initial_seed())
    model = utils.model.lookfor_model_finetune(args)
    print(model)
    # Print random seed state before function call
    print("Random seed state before model initialization:", torch.initial_seed())

    appr = Appr(args)

    # for save classifier layer on each pt model
    if 'only_fc' in args.finetune_type:
        model_state_dict = copy.deepcopy(model.model.state_dict())
        for module in model.model.modules():
            if 'LoRAPiggybackLinear' in str(type(module)):
                module.lora_As[str(args.ft_task)].data.copy_(module.lora_As[str(args.pt_task)].data)
                module.lora_Bs[str(args.ft_task)].data.copy_(module.lora_Bs[str(args.pt_task)].data)
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print('Trainable:', n)

    # Print the number of lora trainable parameters
    n_lora_params = 0
    n_classifier_params = 0
    n_total_params = 0
    for n, p in model.named_parameters():
        if 'lora' not in n and 'classifier' not in n:
            n_total_params += p.numel()
        if 'lora' in n and '.0' in n and p.requires_grad:
            n_lora_params += p.numel()
        if 'classifier' in n and p.requires_grad:
            n_classifier_params += p.numel()
    print(f'Total number of parameters: {n_total_params}')
    print(f'Number of LoRA parameters: {n_lora_params}')
    print(f'Number of classifier parameters: {n_classifier_params}')
    # Print ratio of lora parameters
    print(f'Ratio of LoRA parameters: {n_lora_params/n_total_params}')
    print(f'Ratio of classifier parameters: {n_classifier_params/n_total_params}') 
    
    appr.train(model, accelerator, train_loader, test_loader)
    if 'only_fc' in args.finetune_type:
        torch.save(model.model.classifier, f'only_fc_ft{args.ft_task}_from_pt{args.pt_task}.pt')


if __name__ == '__main__':
    main()

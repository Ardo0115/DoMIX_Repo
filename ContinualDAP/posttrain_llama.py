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

# import utils
import logging
import os
import random
import torch
import datasets
import transformers
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from config import parseing_posttrain
from dataloader.data import get_dataset
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, concatenate_datasets

import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from typing import Any, Dict, List, NewType, Optional, Tuple, Union
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result
class PTDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        if "labels" in batch:
            batch["labels_ori"] = batch["labels"]

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None) # excloude special token
        if self.mlm:
            batch["input_ids"], batch["inputs_ori_ids"],batch["labels"], batch["masked_indices"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        inputs_ori =  inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, inputs_ori, labels, masked_indices


def group_texts(examples,max_seq_length):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result
def get_restaurant_unsup(args):
    data_files = {'train': '/home/ardo0115/data/ContinualDAP/data/yelp_restaurant.txt'}
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{5}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{5}%:]",
    )
    return datasets

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():

    args = parseing_posttrain()




    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args = utils.model.prepare_sequence_posttrain(args)
    # from approaches.posttrain import Appr

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        # Set wandb
        if args.wandb:
            wandb.login()
            wandb.init(project="Continual DAP", name=str(args.baseline)+'_'+str(args.pt_task)+'_'+str(args.dataset_name), config=vars(args))
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            pass
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args.tokenizer = tokenizer
    # Add a new pad token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = prepare_model_for_int8_training(model)
    # Resize the model's embeddings to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))
    # Set up LoRA configuration
    lora_config = LoraConfig(
    r=4,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target specific layers (e.g., query and value projections)
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none"  # No additional bias term  
    )
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)  
    print(model)
    # Freeze all parameters except the scalars in the LoRA layers
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    # Define custom Trainer to train only the scalar parameters
    class TrainerWithScalars(Trainer):
        def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer and scheduler step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            return loss.detach()
    accelerator.wait_for_everyone()

    # if 'comb' in args.baseline:
    #     for t in range(args.pt_task + 1):
    #         if t == 0:
    #             raw_datasets = get_dataset(
    #                 args.data[t], tokenizer=None, args=args)

    #         else:
    #             cur_raw_datasets = get_dataset(
    #                 args.data[t], tokenizer=None, args=args)
    #             train_dataset = cur_raw_datasets["train"]

    #             raw_datasets["train"] = concatenate_datasets(
    #                 [raw_datasets["train"], train_dataset])
    # else:
    #     # Get the dataset
    #     if args.dataset_name is not None:
    #         # Downloading and loading a dataset from the hub.
    #         raw_datasets = get_dataset(
    #             args.dataset_name, tokenizer=None, args=args)
    
    raw_datasets = get_restaurant_unsup(args)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            fn_kwargs={
                'max_seq_length': max_seq_length,
            },
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
    tokenized_datasets = tokenized_datasets.remove_columns(["token_type_ids"])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Check if the dataset is not empty
    if len(tokenized_datasets['train']) == 0:
        raise ValueError("The training dataset is empty. Please check the dataset loading and processing steps.")

    # Example of safely accessing an element
    index = 137244
    if index < len(tokenized_datasets['train']):
        sample = tokenized_datasets['train'][index]
        print(sample)
    else:
        print(f"Index {index} is out of bounds for the dataset size {len(tokenized_datasets['train'])}.")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    # Data collator
    # This one will take care of randomly masking the tokens.
    # data_collator = PTDataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    # Define data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for causal language modeling
    )
    print('train_dataset: ', len(train_dataset))
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(
            range(int(args.max_train_samples)))

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,
        num_workers=0
    )

    train_dataloader_subset_dataset = train_dataset.select(range(int(1e4)))

    train_dataloader_subset = DataLoader(
        train_dataloader_subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=100,
        num_workers=0
    )

    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=5e-5)
    # print trainable params
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
        else:
            print(name, " (frozen)")

    from transformers import get_scheduler

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # print loss per 100 steps
            if progress_bar.n % 10 == 0:
                print(f"Loss: {loss.item()}")
    # # Define training arguments
    # training_args = TrainingArguments(
    #     output_dir="./results_llama",
    #     per_device_train_batch_size=4,
    #     num_train_epochs=1,
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     logging_dir="./logs_llama",
    #     save_total_limit=2,
    #     fp16=True,
    #     # remove_unused_columns=False,
    # )

    # # Initialize the custom trainer
    # # Initialize the Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["validation"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator
    # )

    # # Start training
    # trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine-tuned-llama2-lora-scalars")
    tokenizer.save_pretrained("./fine-tuned-llama2-lora-scalars")



if __name__ == "__main__":
    main()

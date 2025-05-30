import copy
import shutil
import argparse
import logging
import math
import os
import random
import sys
import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from tqdm.auto import tqdm
from networks import prompt
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

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parseing_posttrain():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='roberta-base',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./ckpt',
                        help="Where to store the final model.")
    parser.add_argument("--saved_output_dir", type=str,
                        default='./ckpt', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=111,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=16,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--precision', type=torch.dtype, default=torch.float32)
    parser.add_argument("--baseline",
                        type=str,
                        help="The supported baselines.",
                        choices=["dga", "das", "adapter_hat", "transformer_hat", "adapter_bcl", "adapter_one", "adapter_classic", "prompt_one", "distill", "derpp", "ewc", "ncl", "one", "piggyback", "lora", "lora_piggyback", "piggyback_nonzero", "lora_init", "lora_distill", "lora_piggyback_fixlmhead", "one_comb", "lora_comb"])
    parser.add_argument('--share_weight', action='store_true')
    parser.add_argument(
        "--max_train_samples",
        type=float,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
             "value if set.",
    )
    parser.add_argument("--addition_loss", default='', type=str,
                        help="['kd_origin','cls','replaced','contrastive','kd_generated']")
    parser.add_argument("--sample_type", type=str,
                        default='masked', help="Where to store the final model.")
    parser.add_argument("--classifier_type", type=str,
                        default='both', help="['logits_only','both']")
    parser.add_argument("--teacher_type", type=str,
                        default='naive', help="['cls_token','naive']")
    parser.add_argument("--student_type", type=str,
                        default='naive', help="['cls_token','naive']")
    parser.add_argument("--cls_task", type=str, default='naive',
                        help="['teacher_or_student','fake_or_real']")
    parser.add_argument("--kd_type", type=str, help="['kl','mse',contrastive]")
    parser.add_argument("--kd_layer", type=str,
                        default='output', help="['output','all']")
    parser.add_argument("--kd_on", type=str, default='avg',
                        help="Where to store the final model.")
    parser.add_argument("--temp", type=float, default=0.05,
                        help="Temperature for softmax.")
    parser.add_argument("--pooler_type", type=str,
                        help="What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).")
    parser.add_argument("--contrast_type", type=str,
                        default='naive', help="['naive','add_hard_neg']")
    # https://github.com/princeton-nlp/SimCSE/issues/37
    parser.add_argument("--hard_negative_weight", type=float, default=0)
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument("--layer_to_mask", type=str,
                        help="['head_mask','intermediate_mask','output_mask']")
    parser.add_argument(
        "--masking_threshold",
        default=0.5,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument("--importance_activation",
                        type=str, help="['tanh','sigmoid']")
    parser.add_argument("--idrandom", type=int, help="which sequence to use")
    parser.add_argument("--attention_only", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument(
        "--ppl",
        action="store_true",
        help="whether evaluate on PPL",
    )
    parser.add_argument("--pt_task", type=int, help="task id")
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--base_dir", default='./', type=str, help="task id")
    parser.add_argument("--s", type=int, help="smax")
    parser.add_argument("--smax", default=400, type=int, help="smax")
    parser.add_argument('--thres_cosh', default=50, type=int,
                        required=False, help='(default=%(default)d)')
    parser.add_argument('--thres_emb', default=6, type=int,
                        required=False, help='(default=%(default)d)')
    parser.add_argument('--lamb', type=float, required=False,
                        help='(default=%(default)d)')
    # parser.add_argument("--prune_technique",default='proxy', type=str, help="where to compute the head importance")
    parser.add_argument("--pruning_rate", default=0.8,
                        type=float, help="where to compute the head importance")
    parser.add_argument('--pipline_norm', default='standard_norm',
                        type=str, required=False, help='(default=%(default)d)')
    parser.add_argument('--s_steps', type=int, default=5)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--adv_loss_reg', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--diff_loss_reg', type=float, default=0.1)
    parser.add_argument('--beta', default=0.03, type=float,
                        help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float,
                        help='(default=%(default)f)')
    parser.add_argument('--alpha', default=0.01, type=float,
                        help='(default=%(default)f)')
    parser.add_argument('--semantic_cap_size', default=3,
                        type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--num_semantic_cap', default=3,
                        type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--softmask_compute', type=str)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--reduction_factor", type=float, default=16, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0, type=float)

    # wandb 
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--ngpu', type=int, required=True, help='Number of gpus')
    parser.add_argument(
        "--model_tomerge1",
        type=str,
        default=None,
        help="The name of the model to merge.",
    )
    parser.add_argument(
        "--model_tomerge2",
        type=str,
        default=None,
        help="The name of the model to merge.",
    )
    parser.add_argument(
        "--replay_sample_per_task",
        type=int,
        default=100,
    )
    parser.add_argument("--replay_freq", type=int, default=1)
    parser.add_argument("--replay_alpha", type=float, default=1)
    parser.add_argument("--replay_beta", type=float, default=1)
    args = parser.parse_args()

    return args


def parseing_finetune():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name_or_path',
                        type=str, default='roberta-base')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--params', type=str, default=None)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--least_epoch', type=int)
    parser.add_argument('--patient', default=3, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--precision', type=torch.dtype, default=torch.float32)
    parser.add_argument('--problem_type', type=str,
                        default='single_label_classification')
    # For recording results only ==============================================
    parser.add_argument('--round', type=int,
                        help='the round number in repeated experiments.')
    parser.add_argument("--baseline",
                        type=str,
                        help="The supported baselines.",
                        choices=["dga", "das", "adapter_hat", "adapter_bcl", "adapter_one", "adapter_classic", "prompt_one", "distill", "derpp", "ewc", "ncl", "one", "piggyback", "lora", "piggyback_nonzero", "lora_init", "lora_distill", "das_lora", "lora_piggyback_fixlmhead", "lora_fixlmhead", "one_comb", "dga_one",  "lora_comb", "vanilla"])
    parser.add_argument('--saved_model', type=str)
    parser.add_argument("--saved_output_dir", type=str,
                        default='./ckpt', help="Where to store the final model.")

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument("--generator", default='', type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument("--train_sample_ratio", default=None,
                        type=float, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--n_tokens", type=int, default=100, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides epoch.",
    )
    parser.add_argument("--how_to_block", type=str,
                        help="['grad','head_mask']")
    parser.add_argument("--addition_loss", default='', type=str,
                        help="['kd_origin','cls','replaced','contrastive','kd_generated']")
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--idrandom", type=int, help="which sequence to use")
    parser.add_argument("--pt_task", type=int, help="task id")
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--base_dir", default='./',
                        type=str, help="task id")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("--s", type=int, help="smax")
    parser.add_argument("--smax", default=400, type=int, help="smax")

    parser.add_argument('--beta', default=0.03, type=float,
                        help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float,
                        help='(default=%(default)f)')
    parser.add_argument('--alpha', default=0.01, type=float,
                        help='(default=%(default)f)')
    parser.add_argument('--semantic_cap_size', default=3,
                        type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--num_semantic_cap', default=3,
                        type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--pipline_norm', default='standard_norm',
                        type=str, required=False, help='(default=%(default)d)')
    parser.add_argument('--hyperparameter_tune', type=lambda x: (True if x == 'True' else (False if x ==
                   'False' else argparse.ArgumentTypeError('Boolean value expected'))), default=False)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--sequence_file", type=str, help="sequence file")
    parser.add_argument(
        "--reduction_factor", type=float, default=16, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0, type=float)
    # wandb 
    parser.add_argument('--wandb', action='store_true')

    # Set finetune type
    parser.add_argument("--finetune_type",
                        type=str,
                        help="The supported finetune_type.",
                        required=True,
                        choices=["full", "lora_piggyback", "full_30epoch", "lora", "lora48", "lora8", "train_mergeratio", "train_mergeratio_30epoch", "train_mergeratio_optHype", "full_hypesearch", "train_mergeratioP", "train_mergeratioW", "train_mergeratio_softmax", "train_mergeratioW_optHype_signMerge", "only_fc", "train_mergeratio_only_fc", "train_mergeratioW_signMerge", "train_mergeratio_optHype_clamp", "train_mergeratioW_signMerge_noclamp", "train_mergeratioW_clamp", "train_mergeratio_clamp_eachsignMerge", "train_mergeratioW_signMerge_clamp10", "WsignMerge", "from_foundation", "ModelWise_signMerge", "ModelWise_signMerge_Residual", "full_distill", "ModelWise_signMerge_nosign", "ModelWise_signMerge_noclamp", "ModelWise_signMerge_nosign_noclamp", "ModelWise_signMerge_clamp100", "ModelWise_signMerge_fix1_cut0", "train_mergeratio_onefourth","ModelWise_signMerge_onefourth", "ModelWise_signMerge_effi", "train_mergeratio_effi", "train_mergeratio_onefourth_effi", "train_mergeratio_ModelWise_effi", "train_Mixer", "train_Mixer_initKaiming", "train_Mixer_initKaimingOne", "train_Mixer_onlyTrainMixer", "train_Mixer2", "train_Mixer2ReLU", "train_Mixer_initRand", "train_Mixer_initOtherDiag1e-1", "train_MixerColumn", "train_Mixer_loadFirst", "train_Mixer_loadFirst_initOrtho", "train_Mixer_loadFirst_initKaiming", "train_MixerColumn_loadFirst", "train_MixerColumn_initAllOne", "train_Mixer2_initOrtho", "train_Mixer_onlyTrainB", "train_Mixer_onlyTrainBWeight", "train_Mixer_onlyTrainB_initOtherDiag1e-1", "train_Mixer_onlyTrainBMixer_initOtherDiag1e-1", "train_Mixer_onlyTrainB_initOtherDiag0", "train_Mixer_onlyTrainBMixer_CopySelf", "train_Mixer_onlyTrainBMixer_UniformDiag", "train_Mixer_onlyTrainBMixer", "train_Mixer2_initOrthoUnif", "train_Mixer2_onlyTrainBMixer_initOrthoUnif", "train_Mixer2_onlyTrainBMixer_initOrthoUnifEdit", "train_Mixer_onlyTrainBMixer_UniformDiagAddNoise", "train_Mixer_onlyTrainBMixer_UniformDiag_KaimingB", "train_Mixer_onlyTrainBMixer_UniformDiag_KaimingBMixer", "train_MixerColumn_onlyTrainBMixer_UniformDiag", "train_Mixer_onlyTrainBMixer_UniformDiagEdit", "train_Mixer_onlyTrainBMixer_UniformDiagEdit_KaimingB", "train_MixerColumn_onlyTrainBMixer_UniformDiagEdit", "train_Mixer_onlyTrainB_ZeroedB", "train_Mixer_onlyTrainB_UniformDiagEdit_ZeroedB", "train_Mixer_onlyTrainMixerWeight_UniformDiag", "train_Mixer_onlyTrainMixerWeight_UniformDiagEdit", "train_Mixer_onlyTrainB_UniformDiag_ZeroedB", "train_Mixer_onlyTrainBMixer_UniformDiag_ZeroedB", "train_Mixer_onlyTrainB_UniformDiag", "train_Mixer_onlyTrainB", "train_Mixer_onlyTrainBMixerA_UniformDiag", "train_Mixer_onlyTrainBMixer_UniformDiag_ZeroedB_KaimingA", "train_Mixer_onlyTrainBMixer_initKaiming", "train_Mixer_onlyTrainBMixer_initZero", "train_Mixer_onlyTrainBMixer_UniformDiag_KaimingA", "train_Mixer_onlyTrainAMixer_UniformDiag", "train_Mixer_onlyTrainAB_UniformDiag", "train_Mixer_onlyTrainAB", "train_Mixer_onlyTrainBMixer_UniformDiagOffDiag"])
    parser.add_argument('--ngpu', type=int, required=True, help='Number of gpus')
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        required=True,
        help="Batch size in Pre-training stage (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--model_tomerge1",
        type=str,
        default=None,
        help="The name of the model to merge.",
    )
    parser.add_argument(
        "--model_tomerge2",
        type=str,
        default=None,
        help="The name of the model to merge.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learning rate in Pre-training stage.",
    )
    parser.add_argument('--mergeratio_lr', type=float, default=3e-5)
    parser.add_argument('--Ps_lr', type=float, default=3e-5)
    parser.add_argument('--ortho_reg', type=float, default=0.0)
    parser.add_argument('--optHype', action='store_true')
    parser.add_argument('--WsignMergeTau', type=float, default=None)
    parser.add_argument('--same_epoch_with_DAS', action='store_true')
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--select_LoRA', type=int, default=None)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--MixerDiagInit', type=float, default=None)
    parser.add_argument('--Mixer_lr', type=float, default=None)
    parser.add_argument('--Mixer_weight_decay', type=float, default=None)

    args = parser.parse_args()

    return args

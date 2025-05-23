from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path

import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from networks.baselines.supsup import MultitaskMaskLinear

from networks.transformers.roberta import MyRobertaForSequenceClassification, MyRobertaForMaskedLM
from networks.transformers.roberta_piggyback import PiggybackRobertaForSequenceClassification, PiggybackRobertaForMaskedLM
from networks.transformers.roberta_LoRA import LoRARobertaForSequenceClassification, LoRARobertaForMaskedLM
from networks.prompt.tuning import MyRobertaForSequenceClassificationSoftPromptTunning, MyRobertaForMaskedLMSoftPromptTunning
from networks.posttrain.model import MyModel
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from .manage import copy_weights
import utils
from transformers import (
    MODEL_MAPPING,
    AdamW,
    RobertaConfig,
    RobertaModel,
    get_scheduler,
    Adafactor
)
########################################################################################################################


def print_model_report(model):
    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)

    with open('para', 'a') as clocker_file:
        clocker_file.writelines((human_format(count)).replace('M', '') + '\n')

    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    return


########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


########################################################################################################################

def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean = 0
    std = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean += image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded = mean.view(
        mean.size(0), mean.size(1), 1, 1).expand_as(image)
    for image, _ in loader:
        std += (image - mean_expanded).pow(2).sum(3).sum(2)

    std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()

    return mean, std


########################################################################################################################

# for ACL
def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) /
            res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


def report_val(res):
    # Validation performance
    print(
        ' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
            res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


########################################################################################################################


def cross_entropy(outputs, targets, exp=1, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs)
    tar = torch.nn.functional.softmax(targets)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


########################################################################################################################

def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad = req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad = req_grad
    return


########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


########################################################################################################################

# we need to analysis the results, tensorboard


# default `log_dir` is "runs" - we'll be more specific here
def setup_writer(name):
    writer = SummaryWriter(name)
    return writer


def project_layer(writer, features, class_labels):
    writer.add_embedding(features, metadata=class_labels)


def log_loss(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)


def log_gate(writer, loss_name='log gate', gate_sum_dict=None, global_step=None):
    # ...log the running loss
    writer.add_scalars(loss_name,
                       gate_sum_dict,
                       global_step=global_step)


########################################################################################################################

# distillation ########################

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * \
            (self.T ** 2) / y_s.shape[0]
        return loss


def prepare_sequence_posttrain(args):
    args.sequence_file = 'posttrain'

    if 'dga' in args.baseline:
        args.baseline += '_one'
        args.softmask_compute = 'before_distill'
        args.layer_to_mask = 'head_mask'

    elif 'das' in args.baseline:
        args.softmask_compute = 'before_distill_after_mlm'
        args.layer_to_mask = 'head_mask_intermediate_mask_output_mask'

    sequence_path = f'./sequences/{args.sequence_file}'
    with open(sequence_path, 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    output = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}/{data[args.pt_task]}_roberta/'
    ckpt = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}/{data[args.pt_task-1]}_roberta/'

    if 'dga' in args.baseline or 'das' in args.baseline:
        output_dir = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}'
        args.saved_output_dir = []

        if args.softmask_compute is not None:
            if 'before_distill' in args.softmask_compute and 'one' not in args.baseline:
                for pre_t in range(args.pt_task + 1):
                    new_dir = f'{output}before_distill{pre_t}/'
                    os.makedirs(new_dir, exist_ok=True)
                saved_output = f'{output_dir}/{data[0]}_roberta/before_distill0/'
                # only used in the first one (for the importance of general knoweledge)
                args.saved_output_dir += [saved_output]

            if 'before_distill' in args.softmask_compute and 'one' in args.baseline:
                for pre_t in range(args.pt_task + 1):
                    new_dir = f'{output}before_distill{pre_t}/'
                    os.makedirs(new_dir, exist_ok=True)
                saved_output = f'{output_dir}/{data[args.pt_task]}_roberta/before_distill{args.pt_task}/'
                args.saved_output_dir += [saved_output]

            if 'after_mlm' in args.softmask_compute:
                new_dir = f'{output}after_mlm{pre_t}/'
                os.makedirs(new_dir, exist_ok=True)
                args.saved_output_dir += [
                    f'{output_dir}/{data[t]}_roberta/after_mlm{t}/' for t in range(args.pt_task)]

    else:
        args.saved_output_dir = [
            f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}/{args.dataset_name}/{data[t]}_roberta/' for t in range(args.pt_task + 1)]

    print(
        f'The directory of saved models or saved importances: {args.saved_output_dir}')

    args.output_dir = output
    args.task = args.pt_task

    args.data = data
    args.base_model_name_or_path = "roberta-base"
    args.eval_t = args.pt_task  # we need to use the adapter/plugin

    if 'comb' in args.baseline:
        args.dataset_name = '_unsup'
    else:
        args.dataset_name = data[args.pt_task]

    # no pre-trained for the first
    if args.pt_task == 0 or 'one' in args.baseline or ('wiki' in args.baseline and args.pt_task == 1) or 'lora_comb' in args.baseline:
        args.model_name_or_path = "roberta-base"
    else:
        args.model_name_or_path = ckpt

    if args.eval_only:
        args.model_name_or_path = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}/{data[args.pt_task]}_roberta/'

    print(f'Output directory: {args.output_dir}')
    print(f'Dataset: {args.dataset_name}')
    print(f'Pretrained model: {args.model_name_or_path}')

    if 'ewc' in args.baseline:
        # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000 for ewc
        args.lamb = 5000
    if 'adapter_hat' in args.baseline \
            or 'transformer_hat' in args.baseline \
            or 'adapter_bcl' in args.baseline \
            or 'adapter_classic' in args.baseline:
        args.lamb = 0.75
    args.class_num = 1  # placeholder for pre-training
    return args


def prepare_sequence_finetune(args):
    args.sequence_file = 'posttrain'
    args.base_model_name_or_path = "roberta-base"

    if 'dga' in args.baseline:
        args.baseline += '_one'
        args.softmask_compute = 'before_distill'

    elif 'das' in args.baseline:
        args.softmask_compute = 'before_distill_after_mlm'

    sequence_path = f'./sequences/{args.sequence_file}'
    with open(sequence_path, 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    posttrain2endtask = {"pubmed_unsup": "chemprot_sup", "phone_unsup": "phone_sup", "ai_unsup": "scierc_sup",
                         "camera_unsup": "camera_sup", "acl_unsup": "aclarc_sup", "restaurant_unsup": "restaurant_sup"}
    if 'lora' in args.baseline:
        output = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}/{data[args.pt_task]}_roberta/'
        ckpt = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}_r{args.lora_r}"}/{data[args.pt_task]}_roberta/'
    else:
        output = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}"}/{data[args.pt_task]}_roberta/'
        ckpt = f'{args.base_dir}/seq{args.idrandom}/{args.max_samples}samples/{args.baseline}_pbatch{args.per_device_train_batch_size}_ngpu{args.ngpu}_lr{f"{args.learning_rate:.0e}"}/{data[args.pt_task]}_roberta/'

    args.output_dir = output

    args.dataset_name = posttrain2endtask[data[args.ft_task]]
    if 'vanilla' == args.baseline:
        args.model_name_or_path = "roberta-base"
    else:
        args.model_name_or_path = ckpt

    args.task = args.ft_task

    print(f'Output directory: {args.output_dir}')
    print(f'Dataset: {args.dataset_name}')
    print(f'Pretrained model: {args.model_name_or_path}')
    
    if not args.hyperparameter_tune:
        if args.finetune_type == 'piggyback':
            if args.dataset_name == 'aclarc_sup':
                args.epoch = 20
                args.lr = 3.5e-5
                args.weight_decay = 0
            elif args.dataset_name == 'restaurant_sup':
                args.epoch = 15
                args.lr = 3e-5
                args.weight_decay = 0
            elif args.dataset_name == 'phone_sup':
                args.epoch = 30
                args.lr = 1.5e-5
                args.weight_decay = 0.003
            elif args.dataset_name == 'scierc_sup':
                args.epoch = 25
                args.lr = 4e-5
                args.weight_decay = 0
            elif args.dataset_name == 'chemprot_sup':
                args.epoch = 25
                args.lr = 4e-5
                args.weight_decay = 0
            elif args.dataset_name == 'camera_sup':
                args.epoch = 30
                args.lr = 1.5e-5
                args.weight_decay = 0.003
        elif args.finetune_type == 'piggyback_nonzero':
            if args.dataset_name == 'aclarc_sup':
                args.epoch = 8
                args.lr = 1.2e-5
                args.weight_decay = 0
            elif args.dataset_name == 'restaurant_sup':
                args.epoch = 15
                args.lr = 3e-5
                args.weight_decay = 0
            elif args.dataset_name == 'phone_sup':
                args.epoch = 15
                args.lr = 3e-5
                args.weight_decay = 0.003
            elif args.dataset_name == 'scierc_sup':
                args.epoch = 10
                args.lr = 3e-5
                args.weight_decay = 0
            elif args.dataset_name == 'chemprot_sup':
                args.epoch = 8
                args.lr = 2.3e-5
                args.weight_decay = 0
            elif args.dataset_name == 'camera_sup':
                args.epoch = 15
                args.lr = 3e-5
                args.weight_decay = 0.003
        elif args.finetune_type == 'lora_piggyback':
            # args.epoch = 30
            if args.dataset_name == 'aclarc_sup':
                args.epoch = 10
                args.lr = 0.0000825563281269546
                args.weight_decay = 0.013997436347693602
            elif args.dataset_name == 'restaurant_sup':
                args.epoch = 5
                args.lr = 0.0000562138452426044
                args.weight_decay = 0.02051551571669142
            elif args.dataset_name == 'phone_sup':
                args.epoch = 23
                args.lr = 0.00009916564272175414
                args.weight_decay = 0.04497327262904149
            elif args.dataset_name == 'scierc_sup':
                args.epoch = 34
                args.lr = 0.00001897132489990663
                args.weight_decay = 0.0018982691934866672
            elif args.dataset_name == 'chemprot_sup':
                args.epoch = 8
                args.lr = 0.00005023402861391314
                args.weight_decay = 0.016548869151454117
            elif args.dataset_name == 'camera_sup':
                args.epoch = 16
                args.lr = 0.00009128966095587548
                args.weight_decay = 0.02544224709650889
        elif args.finetune_type == 'full_30epoch':
            args.epoch = 30
        elif 'train_mergeratio' in args.finetune_type:
            if args.optHype:
                args.epoch = 20
                if args.dataset_name == 'restaurant_sup':
                    args.lr = 1e-5
                elif args.dataset_name == 'aclarc_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'scierc_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'phone_sup':
                    args.lr = 1e-5
                elif args.dataset_name == 'chemprot_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'camera_sup':
                    args.lr = 1e-5
                # if args.dataset_name == 'restaurant_sup':
                #     args.epoch = 5
                #     args.lr = 3e-5
                #     # args.mergeratio_lr = 9e-5
                # elif args.dataset_name == 'aclarc_sup':
                #     args.epoch = 30
                #     args.lr = 3e-5
                #     # args.mergeratio_lr = 0.0005
                # elif args.dataset_name == 'scierc_sup':
                #     args.epoch = 30
                #     args.lr = 3e-5
                #     # args.mergeratio_lr = 0.009
                # elif args.dataset_name == 'phone_sup':
                #     args.epoch = 35
                #     args.lr = 1e-5
                #     # args.mergeratio_lr = 0.003
                
                # elif args.dataset_name == 'chemprot_sup':
                #     args.epoch = 35
                #     args.lr = 2e-5
                #     # args.mergeratio_lr = 0.007
                # elif args.dataset_name == 'camera_sup':
                #     args.epoch = 15
                #     args.lr = 1e-5
                #     # args.mergeratio_lr = 0.0003
            else:
                pass
                # if args.dataset_name == 'restaurant_sup':
                #     args.epoch = 5
                # elif args.dataset_name == 'aclarc_sup':
                #     args.epoch = 10
                # elif args.dataset_name == 'scierc_sup':
                #     args.epoch = 5
                # elif args.dataset_name == 'phone_sup':
                #     args.epoch = 15            
                # elif args.dataset_name == 'chemprot_sup':
                #     args.epoch = 10
                # elif args.dataset_name == 'camera_sup':
                #     args.epoch = 15
            
        elif args.finetune_type == 'full':
            if args.optHype:
                args.epoch = 20
                if args.dataset_name == 'restaurant_sup':
                    args.lr = 1e-5
                elif args.dataset_name == 'aclarc_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'scierc_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'phone_sup':
                    args.lr = 1e-5
                elif args.dataset_name == 'chemprot_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'camera_sup':
                    args.lr = 1e-5
            elif args.same_epoch_with_DAS:
                if args.dataset_name in ['aclarc_sup']:
                    args.epoch = 10
                elif args.dataset_name in ["hoc_multi", "scierc_sup", "covidintent_sup", 'restaurant_sup', "laptop_sup"]:
                    args.epoch = 5
                elif args.dataset_name in ['phone_sup', "camera_sup"]:
                    args.epoch = 15
                elif args.dataset_name in ['chemprot_sup', 'rct_sample_sup', 'electric_sup', 'hyperpartisan_sup']:
                    args.epoch = 10
            else:
                pass
            # if args.dataset_name == 'restaurant_sup':
            #     args.epoch = 5
            #     args.lr = 3e-5
            # elif args.dataset_name == 'aclarc_sup':
            #     args.epoch = 30
            #     args.lr = 3e-5
            # elif args.dataset_name == 'scierc_sup':
            #     args.epoch = 30
            #     args.lr = 3e-5
            # elif args.dataset_name == 'phone_sup':
            #     args.epoch = 35
            #     args.lr = 1e-5
            
            # elif args.dataset_name == 'chemprot_sup':
            #     args.epoch = 35
            #     args.lr = 2e-5
            # elif args.dataset_name == 'camera_sup':
            #     args.epoch = 15
            #     args.lr = 1e-5
        else:
            if args.optHype:
                args.epoch = 20
                if args.dataset_name == 'restaurant_sup':
                    args.lr = 1e-5
                elif args.dataset_name == 'aclarc_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'scierc_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'phone_sup':
                    args.lr = 1e-5
                elif args.dataset_name == 'chemprot_sup':
                    args.lr = 3e-5
                elif args.dataset_name == 'camera_sup':
                    args.lr = 1e-5
            
            elif args.same_epoch_with_DAS:
                if args.dataset_name in ['aclarc_sup']:
                    args.epoch = 10
                elif args.dataset_name in ["hoc_multi", "scierc_sup", "covidintent_sup", 'restaurant_sup', "laptop_sup"]:
                    args.epoch = 5
                elif args.dataset_name in ['phone_sup', "camera_sup"]:
                    args.epoch = 15
                elif args.dataset_name in ['chemprot_sup', 'rct_sample_sup', 'electric_sup', 'hyperpartisan_sup']:
                    args.epoch = 10
            else:
                pass

    args.s = args.smax

    return args


def _lookfor_model_prompt(args, training_type):

    if training_type == 'finetune':
        MODEL = MyRobertaForSequenceClassificationSoftPromptTunning
    elif training_type == 'posttrain':
        MODEL = MyRobertaForMaskedLMSoftPromptTunning

    model = MODEL.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        num_labels=args.class_num,
        args=args
    )

    for n, p in model.named_parameters():
        if 'classifier' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if 'one' in args.baseline or args.pt_task == 0:
        model.initialize_soft_prompt(n_tokens=args.n_tokens)

    teacher = model.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        num_labels=args.class_num,
        args=args
    )

    model = MyModel(model, teacher=teacher, args=args)

    return model


def _lookfor_model_adapter(args, training_type):

    if training_type == 'finetune':
        MODEL = MyRobertaForSequenceClassification

    elif training_type == 'posttrain':
        MODEL = MyRobertaForMaskedLM

    model = MODEL.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        num_labels=args.class_num,
        args=args
    )

    if 'one' in args.baseline or args.pt_task == 0:
        model.add_adapter('adapter')  # no mh_adapter by default
    else:
        model.load_adapter(args.model_name_or_path)

    # note this train_adapter will affect even the parent node
    model.train_adapter('adapter')
    # train adapter reopen the adapter

    if 'adapter_classic' in args.baseline:
        for n, p in model.named_parameters():  # nothing is trainable in teacher
            if 'self_attns' in n:  # classic
                p.requires_grad = True

    teacher = MODEL.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        num_labels=args.class_num,
        args=args
    )
    model = MyModel(model, teacher=teacher, args=args)

    return model


def _lookfor_model_piggyback(args, training_type):

    model_pretrained = RobertaModel.from_pretrained(
        args.base_model_name_or_path)
    config = RobertaConfig.from_pretrained(args.base_model_name_or_path)
    if training_type == 'finetune':
        config.baseline = args.baseline
        if config.baseline == 'piggyback':
            config.mask_scale = 1e-2
        elif config.baseline == 'piggyback_nonzero':
            config.mask_scale = 5e-2
            
        model = PiggybackRobertaForSequenceClassification(
            config, args, args.class_num)
        for i in range(args.ft_task + 1):
            model.adaptation(args.class_num, i)

        model_state = torch.load(os.path.join(
            args.model_name_or_path, 'model.pt'), map_location='cpu')
        for n in model_state.keys():
            assert n in model.state_dict()
        model.load_state_dict(model_state, strict=False)

        for n, p in model.named_parameters():
            if 'mask' not in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    elif training_type == 'posttrain':
        config.baseline = args.baseline
        if config.baseline == 'piggyback':
            config.mask_scale = 1e-2
        elif config.baseline == 'piggyback_nonzero':
            config.mask_scale = 5e-2
            
        model = PiggybackRobertaForMaskedLM(
            config, args)
        print(args.model_name_or_path, "\n\n\n\n")
        if "piggyback" in args.model_name_or_path:
            for i in range(args.pt_task + 1):
                model.adaptation(0, i)

            model_state = torch.load(os.path.join(
                args.model_name_or_path, 'model.pt'), map_location='cpu')
            for n in model_state.keys():
                assert n in model.state_dict()
            model.load_state_dict(model_state, strict=False)

        for n, p in model.roberta.named_parameters():
            if 'mask' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    copy_weights(model, model_pretrained)
    model = MyModel(model, teacher=None, args=args)

    return model



def _lookfor_model_lora(args, training_type):

    model_pretrained = RobertaModel.from_pretrained(
        args.base_model_name_or_path)
    config = RobertaConfig.from_pretrained(args.base_model_name_or_path)
    config.training_type = training_type
    if training_type == 'finetune':
        config.finetune_type = args.finetune_type
        config.lora_r = args.lora_r
        config.lora_alpha = args.lora_alpha
        config.lora_dropout = args.lora_dropout
        config.baseline = args.baseline
        config.pt_task = args.pt_task
        config.ft_task = args.ft_task
        config.select_LoRA = args.select_LoRA
        if hasattr(args, 'WsignMergeTau'):
            config.WsignMergeTau = args.WsignMergeTau

        model = LoRARobertaForSequenceClassification(
            config, args, args.class_num)
        for i in range(args.pt_task + 1):
            model.adaptation(args.class_num, i)

        if not args.model_tomerge1:
            model_state = torch.load(os.path.join(
                args.model_name_or_path, 'model.pt'), map_location='cpu')
            for n in model_state.keys():
                if 'lm_head' in n:
                    continue
                assert n in model.state_dict()
            model.load_state_dict(model_state, strict=False)
        else:
            NotImplementedError
            # assert args.model_tomerge2
            # model_state1 = torch.load(os.path.join(
            #     args.model_tomerge1, 'model.pt'), map_location='cpu')
            # model_state2 = torch.load(os.path.join(
            #     args.model_tomerge2, 'model.pt'), map_location='cpu')
            # for n2, p2 in model_state2.items():
            #     if 'lora' in n2 and n2.endswith(f'.{args.pt_task}'):
            #         n1 = '.'.join(n2.split('.')[:-1]) + f'.{args.pt_task-1}'
            #         model_state2[n1] = (0.2*p2 + 0.8*model_state1[n1])
            #         model_state2[n2] = (0.8*p2 + 0.2*model_state1[n1])
                    
            # model.roberta.load_state_dict(model_state2, strict=False)

        
            

        if args.finetune_type == 'lora':
            # for p in model.parameters():
            #     p.requires_grad = True
            for n, p in model.roberta.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif args.finetune_type == 'lora48' or args.finetune_type == 'lora8':
            for n, p in model.roberta.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            import math
            new_r = int(args.finetune_type[4:])
            with torch.no_grad():
                for module_name, module in model.named_modules():
                    if 'LoRAPiggybackLinear' in str(type(module)):
                        if 'lora_comb' in args.baseline:
                            for id in range(1, args.pt_task+1):
                                module.lora_As[str(id)].copy_(module.lora_As[str(0)])
                                module.lora_Bs[str(id)].copy_(module.lora_Bs[str(0)])
                        module.lora_As_before = dict()
                        module.lora_Bs_before = dict()
                        module.lora_As_before[str(args.ft_task)] = module.lora_As[str(args.ft_task)].cuda()
                        module.lora_Bs_before[str(args.ft_task)] = module.lora_Bs[str(args.ft_task)].cuda()
                        module.lora_As_before[str(args.ft_task)].requires_grad = False
                        module.lora_Bs_before[str(args.ft_task)].requires_grad = False

                        module.lora_As[str(args.ft_task)] = nn.Parameter(
                            module.weight.new_zeros((new_r, module.in_features)))
                        module.lora_Bs[str(args.ft_task)] = nn.Parameter(
                            module.weight.new_zeros((module.out_features, new_r)))
                        nn.init.kaiming_uniform_(module.lora_As[str(args.ft_task)], a=math.sqrt(5))
                        nn.init.zeros_(module.lora_Bs[str(args.ft_task)])
                        module.lora_alpha = new_r * 2
                        module.r = new_r
                        module.scaling = module.lora_alpha / (module.r)
                        
                        if 'lm_head' in module_name:
                            module.lora_As[str(args.ft_task)].requires_grad = False
                            module.lora_Bs[str(args.ft_task)].requires_grad = False
                        else:
                            module.lora_As[str(args.ft_task)].requires_grad = True
                            module.lora_Bs[str(args.ft_task)].requires_grad = True
        elif 'ModelWise_signMerge' in args.finetune_type:
            for n, p in model.named_parameters():
                if 'lora' not in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            if 'fix1_cut0' in args.finetune_type:
                for n in model.config.mergeratio_modelwise:
                    if n == str(args.ft_task):
                        model.config.mergeratio_modelwise[n].requires_grad = False
                    else:
                        model.config.mergeratio_modelwise[n].requires_grad = True
            else:
                for n in model.config.mergeratio_modelwise:
                    model.config.mergeratio_modelwise[n].requires_grad = True
            with torch.no_grad():
                for module in model.modules():
                    if 'LoRAPiggybackLinear' in str(type(module)):
                        def T(w):
                            return w.transpose(0, 1) if module.fan_in_fan_out else w
                        if 'Residual' in args.finetune_type:
                            if module.r > 0 and not module.merged:
                                module.weight.data += T(module.lora_Bs[str(module.config.ft_task)] @
                                                    module.lora_As[str(module.config.ft_task)]) * module.scaling
                                module.merged = True
                        module.sign = torch.sign(module.lora_Bs[str(module.config.ft_task)] @ module.lora_As[str(module.config.ft_task)])
                        module.mult_sign = dict()
                        module.to_Merge = dict()
                        for id in range(module.config.pt_task+1):
                            id_sign = torch.sign(module.lora_Bs[str(id)] @ module.lora_As[str(id)])
                            module.mult_sign[id] = ((module.sign * id_sign)>0)
                            module.to_Merge[id] = ((module.mult_sign[id] * (module.lora_Bs[str(id)] @ module.lora_As[str(id)])) * module.scaling).cuda()
            
        elif args.finetune_type == 'lora_piggyback':
            for n, p in model.roberta.named_parameters():
                if 'mask' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif args.finetune_type == 'full' or args.finetune_type == 'WsignMerge' or args.finetune_type == 'from_foundation' or args.finetune_type == 'full_distill':
            for n, p in model.roberta.named_parameters():
                if 'lora' not in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            if 'lora_comb' in args.baseline:
                with torch.no_grad():
                    for module_name, module in model.named_modules():
                        if 'LoRAPiggybackLinear' in str(type(module)):
                            for id in range(1, args.pt_task+1):
                                module.lora_As[str(id)].copy_(module.lora_As[str(0)])
                                module.lora_Bs[str(id)].copy_(module.lora_Bs[str(0)])
        elif 'train_mergeratio' in args.finetune_type:
            if 'only_fc' not in args.finetune_type:
                for n,p in model.roberta.named_parameters():
                    if 'lora' not in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                for n, p in model.named_parameters():
                    if 'classifier' in n or 'ratio' in n or  '11' in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            if 'signMerge' in args.finetune_type:
                with torch.no_grad():
                    for module in model.modules():
                        if 'LoRAPiggybackLinear' in str(type(module)):
                            # def T(w):
                            #     return w.transpose(0, 1) if module.fan_in_fan_out else w
                            # if 'Residual' in args.finetune_type:
                            #     if module.r > 0 and not module.merged:
                            #         module.weight.data += T(module.lora_Bs[str(module.config.ft_task)] @
                            #                             module.lora_As[str(module.config.ft_task)]) * module.scaling
                            #         module.merged = True
                            module.sign = torch.sign(module.lora_Bs[str(module.config.ft_task)] @ module.lora_As[str(module.config.ft_task)])
                            module.mult_sign = dict()
                            for id in range(module.config.pt_task+1):
                                id_sign = torch.sign(module.lora_Bs[str(id)] @ module.lora_As[str(id)])
                                # module.mult_sign = ((module.sign * id_sign)>0).cuda()
                                module.mult_sign[id] = ((module.sign * id_sign)>0).cuda()
            if 'ModelWise' in args.finetune_type:
                model.config.mergeratios_modelwise.requires_grad = True
                if 'effi' in args.finetune_type:
                    with torch.no_grad():
                        for module in model.modules():
                            if 'LoRAPiggybackLinear' in str(type(module)):
                                module.lora_As_stacked = torch.stack([module.lora_As[str(id)] for id in range(args.pt_task+1)], dim=0).cuda()
                                module.lora_Bs_stacked = torch.stack([module.lora_Bs[str(id)] for id in range(args.pt_task+1)], dim=0).cuda()
            else:
                if 'effi' in args.finetune_type:
                    with torch.no_grad():
                        for module in model.modules():
                            if 'LoRAPiggybackLinear' in str(type(module)):
                                module.lora_As_stacked = torch.stack([module.lora_As[str(id)] for id in range(args.pt_task+1)], dim=0).cuda()
                                module.lora_Bs_stacked = torch.stack([module.lora_Bs[str(id)] for id in range(args.pt_task+1)], dim=0).cuda()
                                module.mergeratios = nn.Parameter(torch.ones(args.pt_task+1).cuda())
                                # set one for module.mergeratios[args.ft_task], else zero
                                module.mergeratios.data.fill_(0)
                                module.mergeratios[args.ft_task] = 1
                                module.mergeratios.requires_grad = True
        elif 'train_Mixer' in args.finetune_type:
                import math
                for n,p in model.named_parameters():
                    if 'lora' not in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                with torch.no_grad():
                    for module_name, module in model.named_modules():
                        if 'LoRAPiggybackLinear' in str(type(module)):
                            if 'CopySelf' in args.finetune_type:
                                for id in range(args.pt_task+1):
                                    module.lora_As[str(id)] = module.lora_As[str(args.ft_task)]
                                    module.lora_Bs[str(id)] = module.lora_Bs[str(args.ft_task)]

                            module.lora_As_cat = torch.cat([module.lora_As[str(id)] for id in range(args.pt_task+1)], dim=0).cuda()
                            module.lora_Bs_cat = torch.cat([module.lora_Bs[str(id)] for id in range(args.pt_task+1)], dim=1).cuda()  
                            module.lora_As_cat.requires_grad = False
                            module.lora_Bs_cat.requires_grad = False  
                            if 'KaimingA' in args.finetune_type:
                                nn.init.kaiming_uniform_(module.lora_As_cat, a=math.sqrt(5))
                            if 'KaimingB' in args.finetune_type:
                                nn.init.kaiming_uniform_(module.lora_Bs_cat, a=math.sqrt(5))
                            if 'onlyTrainB' in args.finetune_type or 'onlyTrainAB' in args.finetune_type:
                                if 'ZeroedB' in args.finetune_type:
                                    module.lora_Bs_cat.fill_(0)
                                module.lora_Bs_cat = nn.Parameter(module.lora_Bs_cat)
                                module.lora_Bs_cat.requires_grad = True

                                if 'onlyTrainBMixerA' in args.finetune_type:
                                    module.lora_As_cat = nn.Parameter(module.lora_As_cat)
                                    module.lora_As_cat.requires_grad = True
                            if 'onlyTrainA' in args.finetune_type:
                                module.lora_As_cat = nn.Parameter(module.lora_As_cat)
                                module.lora_As_cat.requires_grad = True
                            if 'Diag' in args.finetune_type and 'initOtherDiag' not in args.finetune_type and 'UniformDiag' not in args.finetune_type:
                                module.lora_Mixer = nn.Parameter(torch.zeros((args.pt_task+1)*module.r).cuda())
                                module.lora_Mixer[(args.ft_task)*module.r:(args.ft_task+1)*module.r].fill_(1)
                                module.lora_Mixer.requires_grad = True
                            else:
                                if 'Mixer2' in args.finetune_type:
                                    # module.r: The rank parameter
                                    # args.pt_task: Number of pre-training tasks (integer)
                                    # args.ft_task: Index of the fine-tuning task (integer starting from 0)

                                    # Compute the size of the matrices
                                    matrix_size = (args.pt_task + 1) * module.r

                                    # Initialize lora_Mixer1 and lora_Mixer2 as zero matrices
                                    module.lora_Mixer1 = nn.Parameter(torch.zeros(matrix_size, matrix_size).cuda())
                                    module.lora_Mixer2 = nn.Parameter(torch.zeros(matrix_size, matrix_size).cuda())

                                    if 'initOrtho' in args.finetune_type:
                                        nn.init.orthogonal_(module.lora_Mixer1)
                                        module.lora_Mixer2.copy_(module.lora_Mixer1.T)
                                        if 'initOrthoUnif' in args.finetune_type:
                                            if 'Edit' in args.finetune_type:
                                                divider = 1.0 / math.sqrt((args.pt_task + 1))
                                            else:
                                                divider = 1.0 / math.sqrt((args.pt_task + 1) * module.r)
                                            module.lora_Mixer1.data *= divider
                                            module.lora_Mixer2.data *= divider
                                    else:
                                        # Create an orthogonal matrix of size (module.r x module.r)
                                        orthogonal_matrix = torch.empty(module.r, module.r).cuda()
                                        nn.init.orthogonal_(orthogonal_matrix)

                                        # Indices for the block corresponding to the fine-tuning task
                                        start_idx = args.ft_task * module.r
                                        end_idx = (args.ft_task + 1) * module.r

                                        # Insert the orthogonal matrix into lora_Mixer1
                                        module.lora_Mixer1.data[start_idx:end_idx, start_idx:end_idx] = orthogonal_matrix

                                        # Initialize the rest of lora_Mixer1 with kaiming_uniform_
                                        with torch.no_grad():
                                            # Create a mask to identify positions not initialized with the orthogonal matrix
                                            mask = torch.ones_like(module.lora_Mixer1.data, dtype=torch.bool)
                                            mask[start_idx:end_idx, start_idx:end_idx] = False  # Exclude the orthogonal block

                                            # Initialize the rest using kaiming_uniform_
                                            nn.init.kaiming_uniform_(module.lora_Mixer1.data[mask].view(-1, 1), a=math.sqrt(5))

                                        # Initialize lora_Mixer2
                                        # Insert the transpose of the orthogonal matrix into lora_Mixer2
                                        module.lora_Mixer2.data[start_idx:end_idx, start_idx:end_idx] = orthogonal_matrix.T

                                        # Zero out the rest of lora_Mixer2 to satisfy the condition M = Identity in the specified block
                                        with torch.no_grad():
                                            mask = torch.ones_like(module.lora_Mixer2.data, dtype=torch.bool)
                                            mask[start_idx:end_idx, start_idx:end_idx] = False
                                            module.lora_Mixer2.data[mask] = 0
                                    module.lora_Mixer1.requires_grad = True
                                    module.lora_Mixer2.requires_grad = True
                                else:
                                    if 'Column' in args.finetune_type:
                                        module.lora_Mixer = nn.Parameter(torch.zeros((args.pt_task+1)*module.r).cuda())
                                    else:
                                        module.lora_Mixer = nn.Parameter(torch.zeros((args.pt_task+1)*module.r, (args.pt_task+1)*module.r).cuda())
                                    if 'initOtherDiag1e-1' in args.finetune_type:
                                        module.lora_Mixer.fill_diagonal_(0.1)
                                    if 'initOtherDiag0' in args.finetune_type:
                                        module.lora_Mixer.fill_diagonal_(0)
                                    if 'initZero' in args.finetune_type:
                                        module.lora_Mixer.fill_(0)
                                    if 'initKaiming' in args.finetune_type and 'loadFirst' not in args.finetune_type:
                                        nn.init.kaiming_uniform_(module.lora_Mixer, a=math.sqrt(5))
                                        if 'initKaimingOne' in args.finetune_type:
                                            module.lora_Mixer[(args.ft_task)*module.r:(args.ft_task+1)*module.r, (args.ft_task)*module.r:(args.ft_task+1)*module.r].fill_diagonal_(1)
                                    else:
                                        if 'Column' in args.finetune_type:
                                            module.lora_Mixer[(args.ft_task)*module.r:(args.ft_task+1)*module.r].fill_(1)
                                        else:
                                            module.lora_Mixer[(args.ft_task)*module.r:(args.ft_task+1)*module.r, (args.ft_task)*module.r:(args.ft_task+1)*module.r].fill_diagonal_(1)
                                        if 'loadFirst' in args.finetune_type:
                                            nn.init.normal_(module.lora_Mixer, mean=0.0, std=1e-5)  # Adjust std as needed
                                            if 'initOrtho' in args.finetune_type:
                                                nn.init.orthogonal_(module.lora_Mixer)
                                                module.lora_Mixer.data *= 1e-5  # Scale down to make initial output negligible
                                            if 'initKaiming' in args.finetune_type:
                                                nn.init.kaiming_uniform_(module.lora_Mixer, a=math.sqrt(5))
                                        if 'initRand' in args.finetune_type:    
                                            # Indices for the block corresponding to the fine-tuning task
                                            start_idx = args.ft_task * module.r
                                            end_idx = (args.ft_task + 1) * module.r
                                            # Introduce minimal randomness to the rest of lora_Mixer
                                            mask = torch.ones_like(module.lora_Mixer.data, dtype=torch.bool)
                                            mask[start_idx:end_idx, start_idx:end_idx] = False  # Exclude the identity block

                                            # Use a scaled-down random initialization
                                            small_std = 1e-5  # Very small standard deviation
                                            module.lora_Mixer.data[mask] = torch.randn(mask.sum(), device=module.lora_Mixer.device) * small_std
                                        if 'initAllOne' in args.finetune_type:
                                            module.lora_Mixer.fill_(1)
                                    module.lora_Mixer.requires_grad = True
                                    if 'onlyTrainB' in args.finetune_type or 'onlyTrainMixerWeight' in args.finetune_type or 'onlyTrainA' in args.finetune_type:
                                        if 'init' not in args.finetune_type:
                                            module.lora_Mixer.fill_(0)
                                            if 'Column' not in args.finetune_type:
                                                module.lora_Mixer.fill_diagonal_(1)
                                        if 'CopySelf' in args.finetune_type or 'UniformDiag' in args.finetune_type:
                                            module.lora_Mixer.fill_(0)
                                            if 'Column' in args.finetune_type:
                                                if 'Edit' in args.finetune_type:
                                                    module.lora_Mixer.fill_(1.0/(args.pt_task+1))
                                                else:
                                                    module.lora_Mixer.fill_(1.0/((args.pt_task+1)*module.r))
                                                if args.MixerDiagInit:
                                                    module.lora_Mixer.fill_(args.MixerDiagInit)
                                            elif 'OffDiag' in args.finetune_type:
                                                def fill_diagonal_with_offset(matrix, value, offset):
                                                    """
                                                    Fill the diagonal with an offset in a matrix with a specified value.
                                                    
                                                    Args:
                                                        matrix (torch.Tensor): The input 2D tensor.
                                                        value (float): The value to fill the diagonal with.
                                                        offset (int): The diagonal offset (+ for upper diagonals, - for lower diagonals).
                                                    """
                                                    rows, cols = matrix.shape
                                                    if offset >= 0:
                                                        # Upper diagonal
                                                        for i in range(min(rows, cols - offset)):
                                                            matrix[i, i + offset] = value
                                                    else:
                                                        # Lower diagonal
                                                        for i in range(min(rows + offset, cols)):
                                                            matrix[i - offset, i] = value
                                                tot_size = (args.pt_task+1)*module.r
                                                for offset in range(-tot_size, tot_size, module.r):
                                                    fill_diagonal_with_offset(module.lora_Mixer, 1.0/((args.pt_task+1)*module.r), offset)
                                                
                                                with torch.no_grad():
                                                    sum_of_all = torch.sum(module.lora_Mixer)
                                                    module.lora_Mixer /= sum_of_all
                                            else:
                                                if 'Edit' in args.finetune_type:
                                                    module.lora_Mixer.fill_diagonal_(1.0/(args.pt_task+1))
                                                else:
                                                    module.lora_Mixer.fill_diagonal_(1.0/((args.pt_task+1)*module.r))
                                                if args.MixerDiagInit is not None:
                                                    module.lora_Mixer.fill_diagonal_(args.MixerDiagInit)
                                            if 'AddNoise' in args.finetune_type:
                                                if 'Edit' in args.finetune_type:
                                                    module.lora_Mixer += torch.randn_like(module.lora_Mixer) * 1e-3
                                                else:
                                                    module.lora_Mixer += torch.randn_like(module.lora_Mixer) * 1e-4
                                        if 'KaimingBMixer' in args.finetune_type:
                                            nn.init.kaiming_uniform_(module.lora_Mixer, a=math.sqrt(5))
                                        if 'onlyTrainBMixer' in args.finetune_type or 'onlyTrainAMixer' in args.finetune_type:
                                            module.lora_Mixer.requires_grad = True
                                        else:
                                            module.lora_Mixer.requires_grad = False
                            if 'NewLoRA' in args.finetune_type:
                                module.new_lora_A = nn.Parameter(torch.zeros(module.r, module.in_features).cuda())
                                module.new_lora_B = nn.Parameter(torch.zeros(module.out_features, module.r).cuda())
                                module.new_lora_A.requires_grad = True
                                module.new_lora_B.requires_grad = True
                                nn.init.kaiming_uniform_(module.new_lora_A, a=math.sqrt(5))

                # if 'onlyTrainMixer_initOrthoUnif' in args.finetune_type:
                #     for n,p in model.named_parameters():
                #         if 'Mixer' in n:
                #             p.requires_grad = True
                #         else:
                #             p.requires_grad = False

                #     for n,p in model.named_parameters():
                #         if 'classifier' in n:
                #             p.requires_grad = True
                
                
                if 'onlyTrainB' in args.finetune_type and 'onlyTrainBWeight' not in args.finetune_type:
                    for n,p in model.named_parameters():
                        if 'lora_Bs_cat' in n:
                            p.requires_grad = True
                        elif 'Mixer' in n:
                            if 'onlyTrainBMixer' in args.finetune_type:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        elif 'lora_As_cat' in n:
                            if 'onlyTrainBMixerA' in args.finetune_type:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        else:
                            p.requires_grad = False
                    for n,p in model.named_parameters():
                        if 'classifier' in n:
                            p.requires_grad = True

                if 'onlyTrainA' in args.finetune_type:
                    for n,p in model.named_parameters():
                        if 'lora_As_cat' in n:
                            p.requires_grad = True
                        elif 'Mixer' in n:
                            if 'onlyTrainAMixer' in args.finetune_type:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        elif 'lora_Bs_cat' in n:
                            if 'onlyTrainAB' in args.finetune_type:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                        else:
                            p.requires_grad = False
                    for n,p in model.named_parameters():
                        if 'classifier' in n:
                            p.requires_grad = True

                if 'onlyTrainBWeight' in args.finetune_type:
                    for n,p in model.named_parameters():
                        if 'lora_Bs_cat' in n:
                            p.requires_grad = True
                    for n,p in model.named_parameters():
                        if 'classifier' in n:
                            p.requires_grad = True

                if 'onlyTrainMixerWeight' in args.finetune_type:
                    for n,p in model.named_parameters():
                        if 'Mixer' in n:
                            p.requires_grad = True
                    for n,p in model.named_parameters():
                        if 'classifier' in n:
                            p.requires_grad = True
                
                if 'NewLoRA' in args.finetune_type:
                    for n,p in model.named_parameters():
                        if 'new_lora' in n:
                            p.requires_grad = True

        elif 'only_fc' in args.finetune_type:
            for n, p in model.named_parameters():
                if 'classifier' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            

    elif training_type == 'posttrain':
        config.lora_r = args.lora_r
        config.lora_alpha = args.lora_alpha
        config.baseline = args.baseline
        config.lora_dropout = args.lora_dropout
        model = LoRARobertaForMaskedLM(config, args)

        if "lora" in args.model_name_or_path:
            for i in range(args.pt_task + 1):
                model.adaptation(0, i)
            model_state = torch.load(os.path.join(
                args.model_name_or_path, 'model.pt'), map_location='cpu')
            for n in model_state.keys():
                assert n in model.state_dict()
            model.load_state_dict(model_state, strict=False)

            for module in model.modules():
                if 'ElementWise' in str(type(module)):
                    print(module.masks['0'].data)

        for n, p in model.roberta.named_parameters():
            if 'lora' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        if 'fixlmhead' in args.baseline:
            for p in model.lm_head.parameters():
                p.requires_grad = False

    copy_weights(model, model_pretrained)
    # if 'train_mergeratio' in args.finetune_type:
        # das_state = torch.load('/home/ardo0115/workspace/piggyback-ContinualLM/seq0/Nonesamples/das_pbatch64_ngpu4_lr1e-04_r8/camera_unsup_roberta/pytorch_model.bin')
    
        # model.load_state_dict(das_state, strict=False)
    teachers = None
    if training_type == 'finetune':
        # for i in range(args.pt_task + 1):
        #     das_state = torch.load(f'/home/ardo0115/workspace/piggyback-ContinualLM/das_classifier_{args.ft_task}_{args.seed}.pt', map_location='cpu')
        #     model.classifier.dense.classifiers[str(i)].weight.data.copy_(das_state['dense.weight'])
        #     model.classifier.dense.classifiers[str(i)].bias.data.copy_(das_state['dense.bias'])
        #     model.classifier.out_proj.classifiers[str(i)].weight.data.copy_(das_state['out_proj.weight'])
        #     model.classifier.out_proj.classifiers[str(i)].bias.data.copy_(das_state['out_proj.bias'])
        if 'distill' in args.finetune_type:
            import copy
            teachers = []
            for i in range(args.pt_task+1):
                teacher = copy.deepcopy(model)
                classifier_state = torch.load(f'/home/ardo0115/workspace/piggyback-ContinualLM/only_fc_ft{args.ft_task}_from_pt{i}.pt', map_location='cpu')
                teacher.classifier.dense.classifiers[str(i)].weight.data.copy_(classifier_state.dense.classifiers[str(args.ft_task)].weight)
                teacher.classifier.dense.classifiers[str(i)].bias.data.copy_(classifier_state.dense.classifiers[str(args.ft_task)].bias)
                teacher.classifier.out_proj.classifiers[str(i)].weight.data.copy_(classifier_state.out_proj.classifiers[str(args.ft_task)].weight)
                teacher.classifier.out_proj.classifiers[str(i)].bias.data.copy_(classifier_state.out_proj.classifiers[str(args.ft_task)].bias)
                teachers.append(teacher.cuda())


    if training_type == 'posttrain':
        das_init = torch.load('/home/ardo0115/workspace/piggyback-ContinualLM/das_init.pt', map_location='cpu')
        shared_keys_butdiff = ['lm_head.layer_norm.bias', 'lm_head.decoder.bias', 'lm_head.bias', 'lm_head.layer_norm.weight']
        unshared_pair = {'lm_head.dense.classifiers.0.weight':'lm_head.dense.weight','lm_head.dense.classifiers.0.bias':'lm_head.dense.bias', 'lm_head.decoder.classifiers.0.weight':'lm_head.decoder.weight', 'lm_head.decoder.classifiers.0.bias': 'lm_head.decoder.bias',
                         'lm_head.dense.classifiers.1.weight':'lm_head.dense.weight','lm_head.dense.classifiers.1.bias':'lm_head.dense.bias', 'lm_head.decoder.classifiers.1.weight':'lm_head.decoder.weight', 'lm_head.decoder.classifiers.1.bias': 'lm_head.decoder.bias',
                         'lm_head.dense.classifiers.2.weight':'lm_head.dense.weight','lm_head.dense.classifiers.2.bias':'lm_head.dense.bias', 'lm_head.decoder.classifiers.2.weight':'lm_head.decoder.weight', 'lm_head.decoder.classifiers.2.bias': 'lm_head.decoder.bias',
                         'lm_head.dense.classifiers.3.weight':'lm_head.dense.weight','lm_head.dense.classifiers.3.bias':'lm_head.dense.bias', 'lm_head.decoder.classifiers.3.weight':'lm_head.decoder.weight', 'lm_head.decoder.classifiers.3.bias': 'lm_head.decoder.bias',
                         'lm_head.dense.classifiers.4.weight':'lm_head.dense.weight','lm_head.dense.classifiers.4.bias':'lm_head.dense.bias', 'lm_head.decoder.classifiers.4.weight':'lm_head.decoder.weight', 'lm_head.decoder.classifiers.4.bias': 'lm_head.decoder.bias',
                         'lm_head.dense.classifiers.5.weight':'lm_head.dense.weight','lm_head.dense.classifiers.5.bias':'lm_head.dense.bias', 'lm_head.decoder.classifiers.5.weight':'lm_head.decoder.weight', 'lm_head.decoder.classifiers.5.bias': 'lm_head.decoder.bias'}

        for n, p in model.named_parameters():
            if n in shared_keys_butdiff:
                p.data.copy_(das_init[n])
            if n in unshared_pair:
                p.data.copy_(das_init[unshared_pair[n]])

    model = MyModel(model, teacher=teachers, args=args)

    return model



def _lookfor_model_others(args, training_type):

    if training_type == 'finetune':
        if args.finetune_type == 'lora':
            config = RobertaConfig.from_pretrained(args.base_model_name_or_path)
            config.training_type = training_type
            config.finetune_type = args.finetune_type
            config.lora_r = args.lora_r
            config.lora_alpha = args.lora_alpha
            config.lora_dropout = args.lora_dropout
            config.baseline = args.baseline
            config.pt_task = args.pt_task
            config.ft_task = args.ft_task
            config.select_LoRA = args.select_LoRA
            MODEL = LoRARobertaForSequenceClassification(
                config, args, args.class_num)
            for i in range(args.pt_task + 1):
                MODEL.adaptation(args.class_num, i)
            
            
        else:
            MODEL = MyRobertaForSequenceClassification

    elif training_type == 'posttrain':
        MODEL = MyRobertaForMaskedLM
    if training_type == 'finetune' and args.finetune_type == 'lora':
        if 'vanilla' in args.baseline:
            model_pretrained = RobertaModel.from_pretrained(
                args.base_model_name_or_path)
            copy_weights(MODEL, model_pretrained)
        else:
            model_state = torch.load(os.path.join(
                args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
            for n in model_state.keys():
                if 'lm_head' in n or 'embeddings.position_ids' in n:
                    continue
                assert n in MODEL.state_dict()
            MODEL.load_state_dict(model_state, strict=False)

        for n, p in MODEL.roberta.named_parameters():
            if 'lora' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        teacher = None
        model = MODEL
    else:
        model = MODEL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            num_labels=args.class_num,
            args=args
        )

        teacher = MODEL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            num_labels=args.class_num,
            args=args
        )
        for param in teacher.parameters():  # nothing is trainable in teacher
            param.requires_grad = False
    # torch.save(model.classifier.state_dict(), f'das_classifier_{args.ft_task}_{args.seed}.pt')
    # exit()
    model = MyModel(model, teacher=teacher, args=args)

    return model


def lookfor_model_posttrain(args):
    if args.model_name_or_path:
        if 'adapter' in args.baseline:
            model = _lookfor_model_adapter(args, 'posttrain')
            return model

        elif 'prompt' in args.baseline:
            model = _lookfor_model_prompt(args, 'posttrain')
            return model

        elif 'piggyback' == args.baseline or 'piggyback_nonzero' == args.baseline:
            model = _lookfor_model_piggyback(args, 'posttrain')
            return model

        elif 'lora' in args.baseline:
            model = _lookfor_model_lora(args, 'posttrain')
            return model

        else:
            model = _lookfor_model_others(args, 'posttrain')
            return model

    else:
        raise ValueError('You must provide the model name or path.')


def lookfor_model_finetune(args):

    if args.model_name_or_path:  # only do electra
        if 'adapter' in args.baseline:
            model = _lookfor_model_adapter(args, 'finetune')
            return model

        elif 'prompt' in args.baseline:
            model = _lookfor_model_prompt(args, 'finetune')
            return model

        elif 'piggyback' == args.baseline or 'piggyback_nonzero' == args.baseline:
            model = _lookfor_model_piggyback(args, 'finetune')
            return model

        elif 'lora' in args.baseline:
            model = _lookfor_model_lora(args, 'finetune')
            return model

        else:
            model = _lookfor_model_others(args, 'finetune')
            return model

    else:
        raise ValueError('You must provide the model name or path.')


def get_view_for(n, p, masks, config, args):
    from utils.roberta import get_view_for as get_view_for

    return get_view_for(n, p, masks, config, args)


def mask(model, accelerator, args):
    from utils.roberta import mask as mask
    return mask(model, accelerator, args)


def get_view_for_tsv(n, model_ori, args):
    from utils.roberta import get_view_for_tsv as get_view_for_tsv

    return get_view_for_tsv(n, model_ori, args)


def lookfor_baseline_variable(self, args):
    from utils.roberta import lookfor_baseline_variable as lookfor_baseline_variable

    return lookfor_baseline_variable(self, args)


def gather_imp(head_imp):
    head_imp_list = [torch.zeros_like(head_imp)
                     for _ in range(dist.get_world_size())]
    # Allgather
    dist.all_gather(tensor_list=head_imp_list, tensor=head_imp.contiguous())

    # Since allgather results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    head_imp_list[dist.get_rank()] = head_imp
    # Get full batch embeddings: (bs x N, hidden)
    head_imp = torch.cat(head_imp_list, 0)

    return head_imp


def gather_mean(head_imp):
    head_importance_list = [torch.zeros_like(
        head_imp) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_importance_list,
                    tensor=head_imp.contiguous())  # everyone need to do this
    head_importance_list = torch.stack(head_importance_list)
    head_importance = torch.mean(head_importance_list, dim=0)
    return head_importance


def frequency_norm(frequency, eps=5e-5):
    # 2D, we need to deal with this for each layer
    frequency = (frequency - frequency.mean()) / (frequency.std()+eps)
    return frequency


def sim_matrix(a, b, eps=1e-8):
    """Batch version of CosineSimilarity."""
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def deep_copy(model, accelerator, args):

    unwrap_model = accelerator.unwrap_model(model)
    unwrap_adaptive_model = deepcopy(unwrap_model)
    optimizer_grouped_parameters = utils.optimize.lookfor_optimize(
        unwrap_adaptive_model, args)  # everything is based on adative_model
    adaptive_optimizer = AdamW(optimizer_grouped_parameters)
    adaptive_model, adaptive_optimizer = accelerator.prepare(
        unwrap_adaptive_model, adaptive_optimizer)

    return adaptive_model, adaptive_optimizer


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, attn_size):
        super(Self_Attn, self).__init__()

        self.query_conv = nn.Linear(attn_size, attn_size)
        self.key_conv = nn.Linear(attn_size, attn_size)
        self.value_conv = nn.Linear(attn_size, attn_size)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B,max_length,hidden_size)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, width, height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, width, height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        # attention =  F.gumbel_softmax(energy,hard=True,dim=-1)
        proj_value = self.value_conv(x).view(
            m_batchsize, width, height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, width, height)

        out = self.gamma*out + x

        return out


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)
        self.position_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, enc_q=None, ranking=None):
        # TODO: Positional/ranking embedding

        if enc_q is None:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_q, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        enc_output = self.layer_norm(enc_output)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5)  # sqrt d_k

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = torch.squeeze(q, 1)
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        if len_q == 1:
            q = q.transpose(1, 2).contiguous().view(sz_b, -1)
        else:
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=40):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, enc_input, ranking):
        return enc_input + self.pos_table[:, ranking].clone().detach()


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(
            dist.get_world_size())]  # TODO: what is a world size?
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class MyContrastive(nn.Module):
    # https://github.com/facebookresearch/moco/blob/3631be074a0a14ab85c206631729fe035e54b525/moco/builder.py#L155
    def __init__(self):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MyContrastive, self).__init__()
        self.n_gpu = torch.cuda.device_count()
        self.T = 1
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.sup_con = SupConLoss()

    def forward(self, x, order_x=None, labels=None, con_type=None):
        """
        labels indicate sample label
        x include all samples to be contrast
        """
        if self.n_gpu > 1:
            x = torch.cat(GatherLayer.apply(x), dim=0)
            if order_x is not None:
                order_x = torch.cat(GatherLayer.apply(order_x), dim=0)
            elif labels is not None:
                labels = torch.cat(GatherLayer.apply(labels), dim=0)

        if con_type == 'supervised':
            loss = self.sup_con(x.unsqueeze(1), labels)
        elif con_type == 'unsupervised':
            loss = self.sup_con(x)
        elif con_type == 'soft_contrast':
            loss = self.unsupervised_loss(x, order_x, labels)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def tacl_loss(z1, z2, contrastive_labels, eps=0.0):
    # https://github.com/yxuansu/TaCL/blob/main/pretraining/bert_contrastive.py
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 1., otherwise 0. I only want masked
    '''
    z1 = torch.cat(GatherLayer.apply(z1), dim=0)
    z2 = torch.cat(GatherLayer.apply(z2), dim=0)
    contrastive_labels = torch.cat(
        GatherLayer.apply(contrastive_labels), dim=0)

    # if self.sim == 'dot_product':
    contrastive_scores = torch.matmul(z1, z2.transpose(1, 2))

    # elif self.sim == 'cosine':  # 'cosine'
    #     masked_rep = masked_rep / masked_rep.norm(dim=2, keepdim=True)
    #     truth_rep = truth_rep / truth_rep.norm(dim=2, keepdim=True)
    #     contrastive_scores = torch.matmul(masked_rep,
    #                                       truth_rep.transpose(1, 2)) / self.temperature  # bsz x seqlen x seqlen
    #
    #
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss = -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss) / contrastive_labels.sum()
    return loss


def taco_loss(z1, z2):
    # https://github.com/yxuansu/TaCL/blob/main/pretraining/bert_contrastive.py
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 1., otherwise 0. I only want masked
    '''
    z1 = torch.cat(GatherLayer.apply(z1), dim=0)
    z2 = torch.cat(GatherLayer.apply(z2), dim=0)

    z1 = z1 / z1.norm(dim=2, keepdim=True)
    z2 = z2 / z2.norm(dim=2, keepdim=True)
    contrastive_scores = torch.matmul(
        z1, z2.transpose(1, 2))  # bsz x seqlen x seqlen

    #
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss = -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen)
    loss = torch.sum(loss)
    return loss


class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1 = D1.view(D1.size(0), -1)
        D1_norm = torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm = D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2 = D2.view(D2.size(0), -1)
        D2_norm = torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm = D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))

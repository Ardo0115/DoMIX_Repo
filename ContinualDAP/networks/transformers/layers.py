import torch
import math
import torch.nn as nn
import avalanche.models as am
import torch.nn.functional as F

from torch.autograd import Variable
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import Optional
from avalanche.benchmarks.scenarios import CLExperience

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, threshold, le):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = le
        outputs[inputs.gt(threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput, None, None
    

class PretrainingMultiTaskClassifier(am.MultiTaskModule):

    def __init__(self, in_features, initial_out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.bias_ = bias
        self.initial_out_features = initial_out_features
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=self.initial_out_features, bias=self.bias_)

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x.to(dtype=torch.float32))


class MultiTaskClassifier(am.MultiTaskModule):

    def __init__(self, in_features, initial_out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.bias = bias
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=num_class, bias=self.bias)

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x)


class ElementWiseLinear(am.MultiTaskModule):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, zero_out=True, bias=True,
                 mask_init='1s', 
                 threshold_fn='binarizer', threshold=None, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = config.mask_scale
        self.mask_init = mask_init
        self.zero_out = zero_out
        self.config = config

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.weight = Parameter(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.masks = nn.ParameterDict({'0': self.make_mask()})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_mask()

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        # Get binarized/ternarized mask from real-valued mask.
        if self.zero_out:
            weight_thresholded = self.get_weight(task_label)
            # Get output using modified weight.
            return F.linear(x, weight_thresholded, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def make_mask(self):
        # Initialize real-valued mask weights.
        mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.

        return Parameter(mask_real)

    def get_weight(self, task_label):
        # For multi-head attention module
        if self.config.baseline == 'piggyback':
            self.mask_thresholded = Binarizer.apply(
                self.masks[str(task_label)], 5e-3, 0)
        elif self.config.baseline == 'piggyback_nonzero':
            self.mask_thresholded = Binarizer.apply(
                self.masks[str(task_label)], 0, -1)

        weight_thresholded = self.mask_thresholded * self.weight

        return weight_thresholded

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)


class LoRALinear(am.MultiTaskModule):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        config=None,
        **kwargs
    ):
        super().__init__()
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        # Mark the weight as unmerged
        self.merged = False
        self.loadFirst = False
        self.merge_weights = merge_weights

        self.weight = Parameter(torch.Tensor(
            out_features, in_features), requires_grad=False)
        self.bias = Parameter(torch.Tensor(out_features), requires_grad=False)

        self.fan_in_fan_out = fan_in_fan_out
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Actual trainable parameters
        if self.r > 0:
            self.lora_As = nn.ParameterDict(
                {'0': nn.Parameter(self.weight.new_zeros((r, in_features)))})
            self.lora_Bs = nn.ParameterDict({'0': nn.Parameter(
                self.weight.new_zeros((out_features, r)))})
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
        else:
            self.lora_As = {}
            self.lora_Bs = {}
        self.init_lora('0')
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self, task_label):
        nn.Linear.reset_parameters(self)
        self.init_lora(task_label)

    def init_lora(self, task_label):
        if hasattr(self, 'lora_As'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_As[task_label], a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs[task_label])

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
    #     nn.Linear.train(self, mode)

    #     for task_label in self.lora_As:
    #         if mode:
    #             if self.merge_weights and self.merged:
    #                 # Make sure that the weights are not merged
    #                 if self.r > 0:
    #                     self.weight.data -= T(self.lora_Bs[task_label] @
    #                                           self.lora_As[task_label]) * self.scaling
    #                 self.merged = False
    #         else:
    #             if self.merge_weights and not self.merged:
    #                 # Merge the weights and mark it
    #                 if self.r > 0:
    #                     self.weight.data += T(self.lora_Bs[task_label] @
    #                                           self.lora_As[task_label]) * self.scaling
    #                 self.merged = True

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.lora_As:
            self.lora_As[str(task_label)] = nn.Parameter(
                self.weight.new_zeros((self.r, self.in_features)))
            self.lora_Bs[str(task_label)] = nn.Parameter(
                self.weight.new_zeros((self.out_features, self.r)))
            self.init_lora(str(task_label))


class LoRAPiggybackLinear(LoRALinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0,
                 fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 config=None,
                 bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None, **kwargs):
        super().__init__(in_features, out_features, r, lora_alpha,
                         lora_dropout, fan_in_fan_out, merge_weights, config, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if self.config.training_type == 'finetune' and self.config.finetune_type == 'lora_piggyback':
            # self.masks_A = nn.ParameterDict({'0': self.make_mask('A')})
            # self.masks_B = nn.ParameterDict({'0': self.make_mask('B')})
            self.masks = nn.ParameterDict({'0': self.make_mask('weight')})
        if self.config.training_type == 'finetune' and 'train_mergeratio' in self.config.finetune_type:
            self.mergeratio = nn.ParameterDict()
            if self.config.finetune_type == 'train_mergeratioP':
                self.Ps = nn.ParameterDict()
    def adaptation(self, num_class, task_label):
        super().adaptation(num_class, task_label)
        if self.config.training_type == 'finetune' and self.config.finetune_type == 'lora_piggyback' and str(task_label) not in self.masks:
            # self.masks_A[str(task_label)] = self.make_mask('A')
            # self.masks_B[str(task_label)] = self.make_mask('B')
            self.masks[str(task_label)] = self.make_mask('weight')
        if self.config.training_type == 'finetune' and 'train_mergeratio' in self.config.finetune_type:
            for task_label in range(self.config.pt_task+1):
                if task_label == self.config.ft_task:
                    self.mergeratio[str(task_label)] = nn.Parameter(torch.Tensor([1.0]))
                else:
                    self.mergeratio[str(task_label)] = nn.Parameter(torch.Tensor([0.0]))
                    if self.config.finetune_type == 'train_mergeratioP':
                        self.Ps[str(task_label)] = nn.Parameter(torch.eye(self.r))
        
                
    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0:
            # result = F.linear(x, T(self.weight), bias=self.bias)
            if self.config.training_type == 'posttrain':
                result = F.linear(x, T(self.weight), bias=self.bias)
                result += (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                           @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
            elif self.config.training_type == 'finetune':
                if self.config.finetune_type == 'lora_piggyback':
                    # thresholded_mask_A = Binarizer.apply(
                    #     self.masks_A[str(task_label)], 5e-3, 0)
                    # thresholded_mask_B = Binarizer.apply(
                    #     self.masks_B[str(task_label)], 5e-3, 0)
                    self.lora_weight = self.weight + (self.lora_Bs[str(task_label)] @ self.lora_As[str(task_label)]) * self.scaling
                    thresholded_mask = Binarizer.apply(
                        self.masks[str(task_label)], 5e-3, 0)
                    result = F.linear(x, T(self.lora_weight * thresholded_mask), bias=self.bias)
                    # result += (self.lora_dropout(x) @ (self.lora_As[str(task_label)] * thresholded_mask_A).transpose(0, 1)
                    #            @ (self.lora_Bs[str(task_label)] * thresholded_mask_B).transpose(0, 1)) * self.scaling
                elif self.config.finetune_type == 'from_foundation':
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    return result
                elif self.config.finetune_type == 'full' or self.config.finetune_type == 'only_fc' or self.config.finetune_type == 'full_distill':
                    if self.r > 0 and not self.merged:
                        if self.config.select_LoRA is not None:
                            self.weight.data += T(self.lora_Bs[str(self.config.select_LoRA)] @
                                            self.lora_As[str(self.config.select_LoRA)]) * self.scaling

                        else:
                            self.weight.data += T(self.lora_Bs[str(task_label)] @
                                            self.lora_As[str(task_label)]) * self.scaling
                        self.merged = True
                    result = F.linear(x, T(self.weight), bias=self.bias)
                elif self.config.finetune_type == 'lora48' or self.config.finetune_type == 'lora8':
                    if self.r > 0 and not self.merged:
                        self.weight.data += T(self.lora_Bs_before[str(task_label)] @
                                            self.lora_As_before[str(task_label)]) * self.scaling
                        self.merged = True
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    result = result + (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                            @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
                elif self.config.finetune_type == 'lora':
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    result = result + (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                            @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
                elif self.config.finetune_type == 'WsignMerge':
                    if self.r > 0 and not self.merged:
                        # result = F.linear(x, T(self.weight), bias=self.bias)
                        # with torch.no_grad():
                        sign = torch.sign(self.lora_Bs[str(self.config.ft_task)] @ self.lora_As[str(self.config.ft_task)])
                        for id in range(self.config.pt_task+1):
                            if id == self.config.ft_task:
                                self.weight.data += T(self.lora_Bs[str(id)] @ self.lora_As[str(id)]) * self.scaling
                            else:
                                id_sign = torch.sign(self.lora_Bs[str(id)] @ self.lora_As[str(id)])
                                mult_sign = (sign * id_sign)>0
                                self.weight.data += self.config.WsignMergeTau * T(mult_sign * (self.lora_Bs[str(id)] @ self.lora_As[str(id)]) * self.scaling)
                        self.merged = True
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    return result
                elif 'ModelWise_signMerge' in self.config.finetune_type:
                    if self.config.finetune_type == 'ModelWise_signMerge':
                        result = F.linear(x, self.weight, bias=self.bias)
                        merged_W = 0
                        for id in range(self.config.pt_task+1):
                            merged_W += torch.clamp(self.config.mergeratio_modelwise[str(id)], min=0, max=1) * (self.to_Merge[id])
                        return result + (self.lora_dropout(x) @ merged_W.t())
                    elif 'effi' in self.config.finetune_type:
                        result = F.linear(x, self.weight, bias=self.bias)
    
                        # Collect matrices from self.to_Merge dictionary up to pt_task
                        to_Merge_list = [self.to_Merge[(id)] for id in range(self.config.pt_task + 1)]
                        
                        # Stack the matrices along a new dimension
                        to_Merge_stacked = torch.stack(to_Merge_list, dim=0)
                        
                        # Get and clamp mergeratio_modelwise values
                        mergeratio_modelwise_stacked = torch.tensor([torch.clamp(self.config.mergeratio_modelwise[str(id)], min=0, max=1) 
                                                                    for id in range(self.config.pt_task + 1)], device=x.device)
                        
                        # Compute merged_W using efficient matrix operations
                        merged_W = torch.einsum('i,ijk->jk', mergeratio_modelwise_stacked, to_Merge_stacked)
                        
                        return result + (self.lora_dropout(x) @ merged_W.t())
                    else:
                        merged_W = 0
                        for id in range(self.config.pt_task+1):
                            # if self.config.finetune_type =='ModelWise_signMerge_nosign':
                            #     merged_W += torch.clamp(self.config.mergeratio_modelwise[str(id)], min=0,max=1) * (self.lora_Bs[str(id)] @ self.lora_As[str(id)])
                            # elif self.config.finetune_type == 'ModelWise_signMerge_noclamp':
                            #     merged_W += (self.config.mergeratio_modelwise[str(id)]) * (self.mult_sign[id] * (self.lora_Bs[str(id)] @ self.lora_As[str(id)]))
                            # elif self.config.finetune_type == 'ModelWise_signMerge_clamp100':
                            #     merged_W += torch.clamp(self.config.mergeratio_modelwise[str(id)], min=0, max=100) * (self.mult_sign[id] * (self.lora_Bs[str(id)] @ self.lora_As[str(id)]))
                            # elif self.config.finetune_type == 'ModelWise_signMerge_nosign_noclamp':
                            #     merged_W += (self.config.mergeratio_modelwise[str(id)]) * (self.lora_Bs[str(id)] @ self.lora_As[str(id)])
                            # else:
                            # merged_W += torch.clamp(self.config.mergeratio_modelwise[str(id)], min=0, max=1) * (self.mult_sign[id] * (self.lora_Bs[str(id)] @ self.lora_As[str(id)]))
                            merged_W += torch.clamp(self.config.mergeratio_modelwise[str(id)], min=0, max=1) * (self.to_Merge[id])
                        
                        return F.linear(x, T(self.weight+ (merged_W)), bias=self.bias)
                elif 'train_mergeratio' in self.config.finetune_type:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    merged_A = 0
                    merged_B = 0
                    merged_W = 0
                    # with torch.no_grad():
                    #     if 'signMerge' in self.config.finetune_type:
                    #         sign = torch.sign(self.lora_Bs[str(self.config.ft_task)] @ self.lora_As[str(self.config.ft_task)])
                    #     if 'eachsignMerge' in self.config.finetune_type:
                    #         signB = torch.sign(self.lora_Bs[str(self.config.ft_task)])
                    #         signA = torch.sign(self.lora_As[str(self.config.ft_task)])
                    if 'effi' in self.config.finetune_type:
                        # Stack lora_As and lora_Bs tensors and mergeratio
                        # lora_As_stacked = torch.stack([self.lora_As[str(id)] for id in range(self.config.pt_task+1)], dim=0)
                        # lora_Bs_stacked = torch.stack([self.lora_Bs[str(id)] for id in range(self.config.pt_task+1)], dim=0)
                        # mergeratio_stacked = torch.tensor([self.mergeratio[str(id)] for id in range(self.config.pt_task+1)], device=x.device)
                        
                        # Compute merged_A and merged_B
                        if 'ModelWise' in self.config.finetune_type:
                            merged_A = torch.einsum('i,ijk->jk', self.config.mergeratios_modelwise, self.lora_As_stacked)
                            merged_B = torch.einsum('i,ijk->jk', self.config.mergeratios_modelwise, self.lora_Bs_stacked)

                        else:
                            merged_A = torch.einsum('i,ijk->jk', self.mergeratios, self.lora_As_stacked)
                            merged_B = torch.einsum('i,ijk->jk', self.mergeratios, self.lora_Bs_stacked)

                        return result + (self.lora_dropout(x) @ merged_A.t() @ merged_B.t()) * self.scaling

                    for id in range(self.config.pt_task+1):
                        if 'train_mergeratioP' in self.config.finetune_type:
                            if id != self.config.ft_task:
                                merged_A += self.mergeratio[str(id)] * self.Ps[str(id)] @ self.lora_As[str(id)]
                                merged_B += self.mergeratio[str(id)] * self.lora_Bs[str(id)] @ torch.inverse(self.Ps[str(id)])
                            else:
                                merged_A += self.mergeratio[str(id)] * self.lora_As[str(id)]
                                merged_B += self.mergeratio[str(id)] * self.lora_Bs[str(id)]
                        elif 'train_mergeratioW' in self.config.finetune_type:
                            if 'signMerge' in self.config.finetune_type:
                                # with torch.no_grad():
                                #     id_sign = torch.sign(self.lora_Bs[str(id)] @ self.lora_As[str(id)])
                                #     mult_sign = (sign * id_sign)>0
                                # if 'noclamp' in self.config.finetune_type:
                                #     merged_W += mult_sign * (self.lora_Bs[str(id)] @ self.lora_As[str(id)])
                                # elif 'clamp10' in self.config.finetune_type:
                                #     merged_W += torch.clamp(self.mergeratio[str(id)], min=0, max=10) * (mult_sign * (self.lora_Bs[str(id)] @ self.lora_As[str(id)]))
                                # elif 'addW' in self.config.finetune_type:
                                #     if self.r > 0 and not self.merged:
                                #         self.weight.data += T(self.lora_Bs[str(id)] @
                                #                             self.lora_As[str(id)]) * self.scaling
                                #         self.merged = True
                                # else:
                                merged_W += torch.clamp(self.mergeratio[str(id)], min=0, max=1) * (self.mult_sign[id] * (self.lora_Bs[str(id)] @ self.lora_As[str(id)]))
                                
                            else:
                                if 'clamp' in self.config.finetune_type:
                                    merged_W += torch.clamp(self.mergeratio[str(id)], min=0, max=1) * self.lora_Bs[str(id)] @ self.lora_As[str(id)]
                                else:
                                    merged_W += self.mergeratio[str(id)] * self.lora_Bs[str(id)] @ self.lora_As[str(id)]
        
                        else:
                            if 'softmax' in self.config.finetune_type:
                                # get softmaxed value of self.mergeratio[str(id)] among all id
                                softmaxed_value = torch.nn.functional.softmax(torch.tensor([self.mergeratio[str(i)] for i in range(self.config.pt_task+1)]), dim=0)
                                merged_A += softmaxed_value[id] * self.lora_As[str(id)]
                                merged_B += softmaxed_value[id] * self.lora_Bs[str(id)]
                            elif 'clamp' in self.config.finetune_type:
                                if 'eachsignMerge' in self.config.finetune_type:
                                    with torch.no_grad():
                                        id_signB = torch.sign(self.lora_Bs[str(id)])
                                        id_signA = torch.sign(self.lora_As[str(id)])
                                        mult_signB = (signB * id_signB)>0
                                        mult_signA = (signA * id_signA)>0
                                    merged_A += torch.clamp(self.mergeratio[str(id)], min=0, max=1) * (mult_signA * self.lora_As[str(id)])
                                    merged_B += torch.clamp(self.mergeratio[str(id)], min=0, max=1) * (mult_signB * self.lora_Bs[str(id)])
                                else:
                                    merged_A += torch.clamp(self.mergeratio[str(id)], min=0, max=1) * self.lora_As[str(id)]
                                    merged_B += torch.clamp(self.mergeratio[str(id)], min=0, max=1) * self.lora_Bs[str(id)]
                            else:
                                merged_A += self.mergeratio[str(id)] * self.lora_As[str(id)]
                                merged_B += self.mergeratio[str(id)] * self.lora_Bs[str(id)]
                    if 'train_mergeratioW' in self.config.finetune_type:
                        # result = F.linear(x, T(self.weight+(merged_W*self.scaling)), bias=self.bias)
                        result = result + (self.lora_dropout(x) @ merged_W.transpose(0, 1)) * self.scaling
                    else:
                        result = result + (self.lora_dropout(x) @ merged_A.transpose(0, 1)
                                @ merged_B.transpose(0, 1)) * self.scaling
                elif 'train_Mixer' in self.config.finetune_type:
                    if 'loadFirst' in self.config.finetune_type:
                        if not self.loadFirst:
                            self.weight.data += T(self.lora_Bs[str(task_label)] @
                                                self.lora_As[str(task_label)]) * self.scaling
                            self.loadFirst = True
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    if 'Mixer2' in self.config.finetune_type:
                        if 'ReLU' in self.config.finetune_type:
                            result += ((((self.lora_dropout(x) @ self.lora_As_cat.T) @ F.relu(self.lora_Mixer1.T)) @ self.lora_Mixer2.T) @ self.lora_Bs_cat.T) * self.scaling
                        else:
                            result += ((((self.lora_dropout(x) @ self.lora_As_cat.T) @ self.lora_Mixer1.T) @ self.lora_Mixer2.T) @ self.lora_Bs_cat.T) * self.scaling

                    else:
                        if 'Column' in self.config.finetune_type:
                            result += (((self.lora_dropout(x) @ self.lora_As_cat.T) * self.lora_Mixer) @ self.lora_Bs_cat.T) * self.scaling
                        else:
                            result += (((self.lora_dropout(x) @ self.lora_As_cat.T) @ self.lora_Mixer.T) @ self.lora_Bs_cat.T) * self.scaling
                    return result
                        

            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
        # nn.Linear.train(self, mode)

    #     for task_label in self.lora_As:
    #         if mode:
    #             if self.merge_weights and self.merged:
    #                 # Make sure that the weights are not merged
    #                 if self.r > 0:
    #                     if self.training_type == 'posttrain':
    #                         self.weight.data -= T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * self.scaling
    #                     elif self.training_type == 'finetune':
    #                         # thresholded_mask_A = Binarizer.apply(
    #                         #     self.masks_A[str(task_label)], 5e-3, 0)
    #                         # thresholded_mask_B = Binarizer.apply(
    #                         #     self.masks_B[str(task_label)], 5e-3, 0)
    #                         # self.weight.data -= T((thresholded_mask_B * self.lora_Bs[task_label]) @ (
    #                         #     thresholded_mask_A * self.lora_As[task_label])) * self.scaling
    #                         thresholded_mask = Binarizer.apply(
    #                             self.masks[str(task_label)], 5e-3, 0)
    #                         self.weight.data -= T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * thresholded_mask * self.scaling
    #                 self.merged = False
    #         else:
    #             if self.merge_weights and not self.merged:
    #                 # Merge the weights and mark it
    #                 if self.r > 0:
    #                     if self.training_type == 'posttrain':
    #                         self.weight.data += T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * self.scaling
    #                     elif self.training_type == 'finetune':
    #                         # thresholded_mask_A = Binarizer.apply(
    #                         #     self.masks_A[str(task_label)], 5e-3, 0)
    #                         # thresholded_mask_B = Binarizer.apply(
    #                         #     self.masks_B[str(task_label)], 5e-3, 0)
    #                         # self.weight.data += T((thresholded_mask_B * self.lora_Bs[task_label]) @ (
    #                         #     thresholded_mask_A * self.lora_As[task_label])) * self.scaling
    #                         thresholded_mask = Binarizer.apply(
    #                             self.masks[str(task_label)], 5e-3, 0)
    #                         print(thresholded_mask)
    #                         self.weight.data += T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * thresholded_mask * self.scaling
    #                 self.merged = True

    def make_mask(self, lora):
        # Initialize real-valued mask weights.
        if lora == 'A':
            mask_real = self.weight.data.new(self.lora_As['0'].size())
        elif lora == 'B':
            mask_real = self.weight.data.new(self.lora_Bs['0'].size())
        elif lora == 'weight':
            mask_real = self.weight.data.new(self.weight.size())

        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.

        return Parameter(mask_real)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)

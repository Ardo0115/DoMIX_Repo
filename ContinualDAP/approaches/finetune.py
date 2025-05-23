
import torch.nn.functional
from utils import utils
from sklearn.metrics import f1_score
import logging
import math
import os
import torch
import wandb
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
import numpy as np
import wandb
import torch.nn.functional as F

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return

    # TODO: Multiple-GPU supprt

    def train(self, model, accelerator, train_loader, test_loader):
        if 'ModelWise_signMerge' in self.args.finetune_type:
            optimizer = AdamW([
                {'params': [p for n,p in model.model.roberta.named_parameters() if ('mergeratio' not in n and p.requires_grad)]},
                {'params': model.model.classifier.parameters()},
                {'params': [model.config.mergeratio_modelwise[n] for n in model.config.mergeratio_modelwise] , 'lr':self.args.mergeratio_lr},
            ], lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif 'ModelWise' in self.args.finetune_type and 'effi' in self.args.finetune_type:
            optimizer = AdamW([
                {'params': [p for n,p in model.model.roberta.named_parameters() if ('mergeratio' not in n and p.requires_grad)]},
                {'params': model.model.classifier.parameters()},
                {'params': model.config.mergeratios_modelwise, 'lr':self.args.mergeratio_lr},
            ], lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            # Set the optimizer
            if 'train_mergeratio' not in self.args.finetune_type:
                if 'Mixer' in self.args.finetune_type:
                    if self.args.Mixer_lr is None:
                        self.args.Mixer_lr = self.args.mergeratio_lr
                    if self.args.Mixer_weight_decay is None:
                        self.args.Mixer_weight_decay = self.args.weight_decay

                    optimizer = AdamW([
                            {'params': [p for n,p in model.model.named_parameters() if ('lora' not in n and p.requires_grad)], 'lr':self.args.lr, 'weight_decay':self.args.weight_decay},
                            {'params': [p for n,p in model.model.named_parameters() if 'lora' in n and 'Mixer' not in n], 'lr':self.args.mergeratio_lr, 'weight_decay':self.args.weight_decay},
                            {'params': [p for n,p in model.model.named_parameters() if 'Mixer' in n], 'lr':self.args.Mixer_lr, 'weight_decay':self.args.Mixer_weight_decay}
                        ])
                else:
                    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                weight_decay=self.args.weight_decay)

            else:
                # Set different lr (==1e-3) for 'mergeratio' params
                if self.args.finetune_type == 'train_mergeratioP':
                    optimizer = AdamW([
                        {'params': [p for n,p in model.model.roberta.named_parameters() if ('mergeratio' not in n and p.requires_grad and 'Ps' not in n)]},
                        {'params': model.model.classifier.parameters()},
                        {'params': [p for n,p in model.model.roberta.named_parameters() if 'mergeratio' in n], 'lr':self.args.mergeratio_lr},
                        {'params': [p for n,p in model.model.roberta.named_parameters() if 'Ps' in n], 'lr':self.args.Ps_lr}
                    ], lr=self.args.lr, weight_decay=self.args.weight_decay)
                

                else:
                    optimizer = AdamW([
                        {'params': [p for n,p in model.model.roberta.named_parameters() if ('mergeratio' not in n and p.requires_grad)]},
                        {'params': model.model.classifier.parameters()},
                        {'params': [p for n,p in model.model.roberta.named_parameters() if 'mergeratio' in n], 'lr':self.args.mergeratio_lr},
                    ], lr=self.args.lr, weight_decay=self.args.weight_decay)


        num_update_steps_per_epoch = math.ceil(
            len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(
                self.args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader)

        logger.info("***** Running training *****")
        logger.info(
            f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset_name}, seed = {self.args.seed}")

        if 'lora' in self.args.baseline:
            os.makedirs(f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}', exist_ok=True)
            summary_path = f'{self.args.output_dir}../finetuneType_{self.args.finetune_type}/{self.args.dataset_name}_finetune_summary'
        else:
            summary_path = f'{self.args.output_dir}../{self.args.dataset_name}_finetune_summary'
        print(f'summary_path: {summary_path}')

        if self.args.early_stop:
            best_train_loss = float('inf')  # Initialize best train loss to infinity
            patience = 2 # Set patience (number of epochs without improvement before stopping)
            epochs_without_improvement = 0  # Track the number of epochs without improvement


        for epoch in range(self.args.epoch):
            if 'onefourth' in self.args.finetune_type and epoch == self.args.epoch//4:
                for n, p in model.model.named_parameters():
                    if 'mergeratio' in n:
                        p.requires_grad = False
            if 'onefourth' in self.args.finetune_type and 'ModelWise_signMerge' in self.args.finetune_type and epoch == self.args.epoch//4:
                for n, p in model.config.mergeratio_modelwise.items():
                    p.requires_grad = False
            if 'onefourth' in self.args.finetune_type and 'ModelWise' in self.args.finetune_type and 'effi' in self.args.finetune_type and epoch == self.args.epoch//4:
                for p in model.config.mergeratios_modelwise:
                    p.requires_grad = False
            print("Epoch {} started".format(epoch))
            train_loader.batch_sampler.sampler.set_epoch(epoch)
            train_acc, training_loss = self.train_epoch(
                model, optimizer, train_loader, accelerator, lr_scheduler)
            
            print("train acc = {:.4f}, training loss = {:.4f}".format(
                train_acc, training_loss))
            if accelerator.is_main_process:
                if self.args.wandb:
                    # log all values
                    wandb.log({"train_acc": train_acc,
                                "training_loss": training_loss})

            if self.args.early_stop:
                # Early stopping logic
                if training_loss < best_train_loss:
                    best_train_loss = training_loss
                    epochs_without_improvement = 0  # Reset the counter if improvement
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
        # lora_dict = {k:v for k,v in model.model.named_parameters() if 'lora' in k}
        # torch.save(lora_dict, f'lora_dict_dir/{self.args.finetune_type}_{self.args.dataset_name}_{self.args.seed}.pt')
        micro_f1, macro_f1, acc, test_loss = self.eval(
            model, test_loader, accelerator)

        if self.args.dataset_name in ['chemprot_sup', 'rct_sample_sup']:
            macro_f1 = micro_f1  # we report micro instead

        logger.info(
            "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(self.args.model_name_or_path,
                                                                                    self.args.dataset_name, macro_f1,
                                                                                    acc, self.args.seed))
        if not self.args.hyperparameter_tune:
            if accelerator.is_main_process:
                os.makedirs(f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}', exist_ok=True)
                progressive_f1_path = f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}/progressive_f1_{self.args.seed}'
                progressive_acc_path = f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}/progressive_acc_{self.args.seed}'

                print(f'Path of progressive f1 score: {progressive_f1_path}')
                print(f'Path of progressive accuracy: {progressive_acc_path}')

                if os.path.exists(progressive_f1_path):
                    f1s = np.loadtxt(progressive_f1_path)
                    accs = np.loadtxt(progressive_acc_path)

                else:
                    f1s = np.zeros(
                        (self.args.ntasks, self.args.ntasks), dtype=np.float32)
                    accs = np.zeros(
                        (self.args.ntasks, self.args.ntasks), dtype=np.float32)

                f1s[self.args.pt_task][self.args.ft_task] = macro_f1
                np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

                accs[self.args.pt_task][self.args.ft_task] = acc
                np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

                if self.args.ft_task == self.args.ntasks - 1:  # last ft task, we need a final one
                    final_f1 = f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}/f1_{self.args.seed}'
                    final_acc = f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}/acc_{self.args.seed}'

                    forward_f1 = f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}/forward_f1_{self.args.seed}'
                    forward_acc = f'{self.args.output_dir}/../finetuneType_{self.args.finetune_type}/forward_acc_{self.args.seed}'

                    print(f'Path of progressive f1 score: {progressive_f1_path}')
                    print(f'Path of progressive accuracy: {progressive_acc_path}')

                    if os.path.exists(progressive_f1_path):
                        f1s = np.loadtxt(progressive_f1_path)
                        accs = np.loadtxt(progressive_acc_path)

                    else:
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[-1][j]) + '\n')
                                f1_file.writelines(str(f1s[-1][j]) + '\n')

                        with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')
                if self.args.wandb:
                    # log all values
                    wandb.log({"macro_f1": macro_f1, "accuracy": acc})

    def train_epoch(self, model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)),
                            disable=not accelerator.is_local_main_process)
        model.train()
        train_acc = 0.0
        training_loss = 0.0
        total_num = 0.0
        if self.args.ortho_reg > 0:
            eye_dict = {}
            for module in model.model.modules():
                if 'Linear' in str(type(module)):
                    eye_dict[module] = torch.eye(module.weight.size(1)).to(module.weight.device)
        for batch, inputs in enumerate(dataloader):
            if 'transformer_hat' in self.args.baseline:
                model_ori = accelerator.unwrap_model(model)
                head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                res = model.model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance)
            elif 'piggyback' in self.args.baseline or 'lora' in self.args.baseline:
                res = model.model(**inputs, task_label=self.args.ft_task, return_dict=True)
                if 'distill' in self.args.finetune_type:
                    with torch.no_grad():
                        teachers_logits = []
                        for task_id in range(self.args.pt_task+1):
                            # teacher_res = model.teacher[task_id](**inputs, task_label=task_id, return_dict=True)
                            # teachers_logits.append(teacher_res.logits)
                            # all one tensors
                            teachers_logits.append(torch.ones_like(res.logits))
                        logits_stack = torch.stack(teachers_logits, dim=0)
                        softmax_values = F.softmax(logits_stack, dim=2)
                        max_indices = torch.argmax(softmax_values.max(dim=2).values, dim=0)
                        most_confident_logits = logits_stack[max_indices, torch.arange(logits_stack.size(1))]
            elif 'lora' in self.args.finetune_type:
                res = model.model(**inputs, task_label=self.args.ft_task, return_dict=True)

            else:
                res = model.model(**inputs, return_dict=True)

            outp = res.logits
            loss = res.loss
            if self.args.ortho_reg > 0:
                ortho_loss = 0
                for module in model.model.modules():
                    if 'Linear' in str(type(module)):
                        ortho_loss += self.args.ortho_reg * torch.norm(module.weight.T @ module.weight - eye_dict[module])
                loss += ortho_loss
            if 'distill' in self.args.finetune_type:
                temperature = 2
                soft_targets = F.softmax(most_confident_logits / temperature, dim=1)
                current_probs = F.log_softmax(outp / temperature, dim=1)
                distillation_loss = F.kl_div(current_probs, soft_targets, reduction='batchmean') * (temperature**2)
                distill_lamb = 0.5
                loss += distill_lamb * distillation_loss
            optimizer.zero_grad()
            accelerator.backward(loss)

            
                
            
            if 'lora_piggyback' == self.args.finetune_type:
                for module in model.model.modules():
                    if 'Piggyback' in str(type(module)):
                        # abs_weights_A = module.lora_As[str(self.args.ft_task)].data.abs()
                        # abs_weights_B = module.lora_Bs[str(self.args.ft_task)].data.abs()
                        # module.masks_A[str(self.args.ft_task)].grad.data.div_(
                        #     abs_weights_A.mean())
                        # module.masks_B[str(self.args.ft_task)].grad.data.div_(
                        #     abs_weights_B.mean())
                        abs_lora = module.lora_weight.data.abs()
                        module.masks[str(self.args.ft_task)].grad.data.div_(
                            abs_lora.mean())

            # if batch == 0:
            #     for n, p in accelerator.unwrap_model(model).named_parameters():
            #         if p.grad is not None:
            #             print('n,p: ', n)

            optimizer.step()
            lr_scheduler.step()

            pred = outp.max(1)[1]

            predictions = accelerator.gather(pred)
            references = accelerator.gather(inputs['labels'])

            train_acc += (references == predictions).sum().item()
            training_loss += loss.item()
            total_num += references.size(0)

            progress_bar.update(1)
            
            if self.args.wandb:
                if batch % 10 == 0 and accelerator.is_local_main_process:
                    if self.args.ortho_reg > 0:
                        wandb.log({"loss_iter": loss.item(), "lr": optimizer.param_groups[0]['lr'], "ortho_loss": ortho_loss.item()})
                    elif 'distill' in self.args.finetune_type:
                        wandb.log({"loss_iter": loss.item(), "lr": optimizer.param_groups[0]['lr'], "distill_loss": distillation_loss.item()})
                    else:
                        wandb.log({"loss_iter": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                    if 'ModelWise_signMerge' in self.args.finetune_type:
                        # log all config.mergeratio_modelwise
                        for n, p in model.config.mergeratio_modelwise.items():
                            wandb.log({f"mergeratio_{n}": p.item()})
                    if 'ModelWise' in self.args.finetune_type and 'effi' in self.args.finetune_type:
                        for i, p in enumerate(model.config.mergeratios_modelwise):
                            wandb.log({f"mergeratio_{i}": p.item()})

            # break
        if 'fix1_cut0' in self.args.finetune_type:
            for n in model.config.mergeratio_modelwise:
                if model.config.mergeratio_modelwise[n] < 0:
                    model.config.mergeratio_modelwise[n].requires_grad = False

        return train_acc / total_num, training_loss / total_num

    def eval(self, model, dataloader, accelerator):
        model.eval()
        label_list = []
        prediction_list = []
        total_loss = 0
        total_num = 0
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)),
                            disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']
                if 'piggyback' in self.args.baseline or 'lora' in self.args.baseline:
                    res = model.model(**inputs, task_label=self.args.ft_task, return_dict=True)
                elif 'lora' in self.args.finetune_type:
                    res = model.model(**inputs, task_label=self.args.ft_task, return_dict=True)
                else:
                    res = model.model(**inputs, return_dict=True)

                real_b=input_ids.size(0)
                loss = res.loss
                outp = res.logits
                if self.args.problem_type != 'multi_label_classification':
                    pred = outp.max(1)[1]
                else:
                    pred = outp.sigmoid() > 0.5

                total_loss += loss.data.cpu().numpy().item()*real_b
                total_num += real_b

                predictions = accelerator.gather(pred)
                references = accelerator.gather(inputs['labels'])

                label_list += references.cpu().numpy().tolist()  # we may use multi-node
                prediction_list += predictions.cpu().numpy().tolist()
                progress_bar.update(1)
                # break

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        accuracy = sum([float(label_list[i] == prediction_list[i])
                       for i in range(len(label_list))]) * 1.0 / len(prediction_list)
        
        if self.args.wandb:
            wandb.log({"Eval_Loss/Task%s" % (self.args.ft_task): total_loss/total_num,
                "Eval_Acc/Task%s" % (self.args.ft_task): accuracy,
                "Eval_Micro_F1/Task%s" % (self.args.ft_task): micro_f1,
                "Eval_Macro_F1/Task%s" % (self.args.ft_task): macro_f1, })

        return micro_f1, macro_f1, accuracy, total_loss/total_num

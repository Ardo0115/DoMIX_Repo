from networks.baselines import ewc, hat, softmask, memory
import utils
import logging
import torch
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def compute(self, model, head_impt, intermediate_impt, output_impt, batch, loss, buffer, mask_back, outputs, epoch, step, accelerator, replay_outputs=None, replay_batch_tmp=None):

    if 'derpp' in self.args.baseline \
            and not (buffer is None or buffer.is_empty()) \
            and step % self.args.replay_freq == 0:

        # replay_batch = buffer.get_data(size=batch['input_ids'].shape[0])
        # replay_batch_tmp = {
        #     'input_ids': replay_batch[0],
        #     'inputs_ori_ids': replay_batch[0],
        #     'attention_mask': replay_batch[1],
        #     'labels': replay_batch[2],
        #     'logits': replay_batch[3].float(),
        #     'task': replay_batch[4]
        # }
        # # Ensure no tensors are inadvertently modified in place
        # replay_batch_tmp = {k: v.clone() for k, v in replay_batch_tmp.items()}

        # # print replay_batch_tmp info
        # for k, v in replay_batch_tmp.items():
        #     print(f'{k}: {v.shape}')

        # # Pass the replay batch to the model with required flags
        # replay_outputs = model(replay_batch_tmp, derpp_replay=True)
        loss = loss + (replay_outputs.loss * self.args.replay_beta)
        mse_loss = torch.nn.MSELoss()
        loss = loss + (mse_loss(
            replay_outputs.hidden_states[-1].float(), replay_batch_tmp['logits'].float().clone()) * self.args.replay_alpha).float()

    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        contrast_loss = outputs.contrast_loss
        loss = loss + contrast_loss
    if 'distill' in self.args.baseline:
        distill_loss = outputs.distill_loss
        loss = loss + distill_loss
    if 'simcse' in self.args.baseline:
        simcse_loss = outputs.simcse_loss
        loss = loss + simcse_loss
    if 'tacl' in self.args.baseline:
        tacl_loss = outputs.tacl_loss
        loss = loss + tacl_loss
    if 'taco' in self.args.baseline:
        taco_loss = outputs.taco_loss
        loss = loss + taco_loss
    if 'infoword' in self.args.baseline:
        infoword_loss = outputs.infoword_loss
        loss = loss + infoword_loss

    loss = loss / self.args.gradient_accumulation_steps


    # # Enable anomaly detection if needed to debug further
    # torch.autograd.set_detect_anomaly(True)
    accelerator.backward(loss)

    if accelerator.is_main_process and epoch < 1 and step < 1:
        for n, p in accelerator.unwrap_model(model).named_parameters():
            if p.grad is not None:
                print(
                    f'Gradient of param "{n}" with size {tuple(p.size())} detected')

    if self.args.pt_task > 0 and \
            ('adapter_hat' in self.args.baseline
             or 'transformer_hat' in self.args.baseline
             or 'adapter_bcl' in self.args.baseline
             or 'adapter_classic' in self.args.baseline):
        for n, p in model.named_parameters():
            if n in mask_back and p.grad is not None:
                p.grad.data *= mask_back[n]

    if 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        # Compensate embedding gradients
        for n, p in model.named_parameters():
            if ('adapters.e' in n or 'model.e' in n) and p.grad is not None:
                num = torch.cosh(torch.clamp(self.args.s * p.data, -self.args.thres_cosh,
                                             self.args.thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.args.smax / self.args.s * num / den

    # we need this even for the first task
    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        softmask.soft_mask_gradient(model, head_impt, intermediate_impt, output_impt, accelerator, epoch, step,
                                    self.args)

    if 'piggyback' == self.args.baseline or 'piggyback_nonzero' == self.args.baseline:
        for module in model.modules():
            if 'ElementWise' in str(type(module)):
                abs_weights = module.weight.data.abs()
                module.masks[str(self.args.pt_task)].grad.data.div_(
                    abs_weights.mean())

    return model

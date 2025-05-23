from copy import deepcopy
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


def compute(self, model, batch, head_impt, intermediate_impt, output_impt, self_fisher, mask_pre, train_loader, step, accelerator, buffer=None):

    if not 'lora' in self.args.baseline:
        self.args.s = (self.args.smax - 1 / self.args.smax) * step / \
            len(train_loader) + 1 / self.args.smax  # Only for HAT based model

    if 'ewc' in self.args.baseline:
        outputs = model(batch, self_fisher=self_fisher)
    elif 'adapter_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        masks = self.mask(model, accelerator, self.args)
        outputs = model(batch, masks=masks, mask_pre=mask_pre)
    elif 'transformer_hat' in self.args.baseline:
        model_ori = accelerator.unwrap_model(model)
        head_importance, intermediate_importance, output_importance = model_ori.model.transformer_mask()
        masks = self.mask(model, accelerator, self.args)  # need mask
        outputs = model(batch, head_mask=head_importance,
                        intermediate_mask=intermediate_importance, output_mask=output_importance,
                        masks=masks, mask_pre=mask_pre)

    elif 'dga' in self.args.baseline or 'das' in self.args.baseline:
        outputs = model(batch,
                        head_mask=head_impt,
                        intermediate_mask=intermediate_impt,
                        output_mask=output_impt)
    elif 'piggyback' in self.args.baseline or 'lora' in self.args.baseline:
        if 'lora_comb' in self.args.baseline:
            outputs = model(batch, task_label=0)
        else:
            outputs = model(batch, task_label=self.args.pt_task)
    elif 'derpp' in self.args.baseline \
            and not (buffer is None or buffer.is_empty()) \
            and step % self.args.replay_freq == 0:
        outputs = model(batch)
        replay_batch = buffer.get_data(size=batch['input_ids'].shape[0])
        replay_batch_tmp = {
            'input_ids': replay_batch[0],
            'inputs_ori_ids': replay_batch[0],
            'attention_mask': replay_batch[1],
            'labels': replay_batch[2],
            'logits': replay_batch[3].float(),
            'task': replay_batch[4]
        }
        # Ensure no tensors are inadvertently modified in place
        replay_batch_tmp = {k: v.clone() for k, v in replay_batch_tmp.items()}

        # Pass the replay batch to the model with required flags
        replay_outputs = model(replay_batch_tmp, derpp_replay=True)
        return self, model, outputs, replay_outputs, replay_batch_tmp
    else:
        outputs = model(batch)
    return self, model, outputs

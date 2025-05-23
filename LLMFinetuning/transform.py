import torch
import os
from tqdm import tqdm
import transformers
import matplotlib.pyplot as plt


def get_transformed_weights(ref_t, moving_t, lr=1e-3, num_warmup_steps=1000, num_training_steps=10000, ignore_params=[]):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    to_optimize = {}
    for n, p in ref_t.items():
        ref_t[n] = p.cuda()
        moving_t[n] = moving_t[n].cuda()

    for n_ref,p_ref in tqdm(ref_t.items()):
        if 'lora_B' in n_ref:
            continue
        n_A = n_ref
        n_B = n_ref.replace('lora_A', 'lora_B')

        P = torch.eye(p_ref.shape[0]).cuda()
        P.requires_grad = True
        to_optimize[n_A] = {'P': P}
    
    optimizer  = torch.optim.Adam([to_optimize[n]['P'] for n in to_optimize], lr=lr)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=1)
    loss_history = []

    for i in tqdm(range(num_training_steps)):
        optimizer.zero_grad()
        loss = 0
        for n_A in to_optimize:
            n_B = n_A.replace('lora_A', 'lora_B')
            P = to_optimize[n_A]['P']
            B1 = ref_t[n_B]
            B2 = moving_t[n_B]
            A1 = ref_t[n_A]
            A2 = moving_t[n_A]

            loss += torch.norm(B1.t() @ B2 @ torch.inverse(P))**2
            loss += torch.norm(A1 @ (P @ A2).t())**2
        if i % 10 == 0:
            print(f'Iter {i} Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
    

    for n_A in to_optimize:
        if 'lora_B' in n:
            continue
        P = to_optimize[n_A]['P']
        moving_t[n_A] = P @ moving_t[n_A]
        moving_t[n_B] = moving_t[n_B] @ torch.inverse(P)
    
    return moving_t, loss_history

    


if __name__ == '__main__':        

    ref_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/math_lora_r=8'
    moving_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/lora_r=8'

    ref_lora = torch.load(os.path.join(ref_dir, 'adapter_model.bin'), map_location='cpu')
    moving_lora = torch.load(os.path.join(moving_dir, 'adapter_model.bin'), map_location='cpu')

    moved_st, loss_history = get_transformed_weights(ref_lora, moving_lora, lr=1e-2, num_warmup_steps=100, num_training_steps=1000)

    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig('loss_history_ABZero.png')

    torch.save(moved_st, os.path.join(moving_dir, 'adapter_model_ABZero_Reflora_r=8.bin'))




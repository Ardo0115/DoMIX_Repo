import torch
import os
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def trainclose_lerp(t1, n_seen_domains, lr=1e-3, num_warmup_steps=1000, num_training_steps=10000):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    orig_dist = 0
    to_optimize = {}
    # move all params to cuda
    for n, p in tqdm(t1.items()):
        t1[n] = p.cuda()

    for n, p in tqdm(t1.items()):
        if 'lora_As.0' in n:    # n : query.lora_As.0
            for i in range(0, n_seen_domains):
                n2 = n.replace('lora_As.0', f'lora_As.{i}')
                E = torch.zeros(p.shape[0], p.shape[0]).to(p.device)
                E.requires_grad = True
                to_optimize[n2] = {'E': E}
    # Function to compute the matrix exponential
    def matrix_exponential(A):
        return torch.matrix_exp(A)
    optimizer = torch.optim.Adam([to_optimize[n]['E'] for n in to_optimize], lr=lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    loss_history = []
    for s in tqdm(range(num_training_steps)):
        optimizer.zero_grad()
        loss = 0
        A_reg_loss = 0
        B_reg_loss = 0
        sim_loss = 0
        for n in to_optimize:
            if 'lora_As.0' in n:
                # n1 = n
                # n2 = '.'.join(n.split('.')[:-1]) + f'.{moving_idx}'
                A_name_list = []
                B_name_list = []
                for i in range(0, n_seen_domains):
                    A_name_list.append(n.replace('lora_As.0', f'lora_As.{i}'))
                    B_name_list.append(n.replace('lora_As.0', f'lora_Bs.{i}'))

                E_list = [to_optimize[A_name]['E'] for A_name in A_name_list]
                A_list = [t1[A_name] for A_name in A_name_list]
                B_list = [t1[B_name] for B_name in B_name_list]

                PA_norm_list = []
                BP_inv_norm_list = []

                for E, A, B in zip(E_list, A_list, B_list):
                    P = matrix_exponential(E)
                    PA = P @ A
                    norm_diff = ((PA.norm(dim=1) - A.norm(dim=1))**2).sum()
                    reg_term = 0.1 * norm_diff
                    loss += reg_term
                    A_reg_loss += reg_term

                    P_inv = torch.inverse(P)
                    BP_inv = B @ P_inv

                    norm_diff = ((BP_inv.norm(dim=0) - B.norm(dim=0))**2).sum()
                    reg_term = 0.1 * norm_diff
                    loss += reg_term
                    B_reg_loss += reg_term

                    PA_norm = PA / (PA.norm(dim=1, keepdim=True) + 1e-8)

                    PA_norm_list.append(PA_norm)

                    BP_inv_norm = BP_inv / (BP_inv.norm(dim=0, keepdim=True) + 1e-8)

                    BP_inv_norm_list.append(BP_inv_norm)

                for i in range(0, n_seen_domains-1):
                    for j in range(i+1, n_seen_domains):
                        cos_sim_matrix = PA_norm_list[i] @ PA_norm_list[j].T
                        loss += ((cos_sim_matrix-1) ** 2).sum()
                        sim_loss += ((cos_sim_matrix-1) ** 2).sum()
                
        if s % 10 == 0:
            print('loss : ', loss.item())
            print('A_reg_loss : ', A_reg_loss.item())
            print('B_reg_loss : ', B_reg_loss.item())
            print('sim_loss : ', sim_loss.item())
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    print('total distance : ', orig_dist)

    for n in t1:
        if 'lora_As.0' in n:
            # n1 = n
            # n2 = '.'.join(n.split('.')[:-1]) + f'.{moving_idx}'
            A_name_list = []
            B_name_list = []
            for i in range(0, n_seen_domains):
                A_name_list.append(n.replace('lora_As.0', f'lora_As.{i}'))
                B_name_list.append(n.replace('lora_As.0', f'lora_Bs.{i}'))

            E_list = [to_optimize[A_name]['E'] for A_name in A_name_list]
            A_list = [t1[A_name] for A_name in A_name_list]
            B_list = [t1[B_name] for B_name in B_name_list]

            for A_name, B_name, E, A, B in zip(A_name_list, B_name_list, E_list, A_list, B_list):
                P = matrix_exponential(E)
                t1[A_name] = P @ A
                t1[B_name] = B @ torch.inverse(P)


    return t1, loss_history

if __name__ == '__main__':
    # parse argumetns
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_model', type=int, choices=[0, 1, 2, 3, 4, 5], required=True)
    parser.add_argument('--architecture', type=str, required=True, choices=['roberta-base', 't5-base', 'Undi95/Meta-Llama-3-8B-hf'])
    # parser.add_argument('--to_save_directory', type=str, required=True )

    args = parser.parse_args()

    splitted_arch = str(args.architecture).split('-')[0]
    dirs = [f'restaurant_unsup_{splitted_arch}', f'acl_unsup_{splitted_arch}', f'ai_unsup_{splitted_arch}', f'phone_unsup_{splitted_arch}', f'pubmed_unsup_{splitted_arch}', f'camera_unsup_{splitted_arch}']
    ref_dir = dirs[args.fixed_model]
    

    # make directory f'./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8 if not exist
    if not os.path.exists(f'./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8'):
        os.makedirs(f'./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')
    model_tomerge1 = f'./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir}/'
    model_state1 = torch.load(os.path.join(model_tomerge1, 'model.pt'), map_location='cpu')
    if 'Llama' in args.architecture:
        for n in list(model_state1.keys()):
            model_state1[n[17:-7]+'s.'+str(args.fixed_model)] = model_state1.pop(n)
    os.system(f'cp -a {model_tomerge1} ./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')

    
    # copy directory of model_tomerge1
    # os.system(f'rsync -av --exclude="model.pt" ./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/camera_unsup_{splitted_arch} ./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')


    model_state1, loss_history = trainclose_lerp(model_state1, args.fixed_model+1, lr=1e-2, num_warmup_steps=50, num_training_steps=500)

    # plot the loss history
    #save loss_hitory plot
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig('loss_history_ortho2.png')

    torch.save(model_state1, f'./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir}/model.pt')
    
    # os.system(f'mv ./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir} ./New_lora_transform_ACloseCossimEachOther{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/camera_unsup_{splitted_arch}')




            






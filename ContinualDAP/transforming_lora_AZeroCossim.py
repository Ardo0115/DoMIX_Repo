import torch
import os
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def trainclose_lerp(t1, t2, lr=1e-3, num_warmup_steps=1000, num_training_steps=10000, ignore_params=[], ref_idx = None, moving_idx = None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    before_dist = 0
    orig_dist = 0
    merged_params1 = {}
    merged_params2 = {}
    to_optimize = {}
    # move all params to cuda
    for n, p in tqdm(t1.items()):
        t1[n] = p.cuda()
    for n, p in tqdm(t2.items()):
        t2[n] = p.cuda()

    for n2, p2 in tqdm(t2.items()):
        if 'lora_As' in n2 and n2.endswith(f'.{moving_idx}'):    # n2 : query.lora_As.1
            n1 = '.'.join(n2.split('.')[:-1]) + f'.{ref_idx}' # n1 : query.lora_As.0
            n1_B = n1.replace('lora_As', 'lora_Bs')     # n1_B : query.lora_Bs.0
            n2_B = n2.replace('lora_As', 'lora_Bs')     # n2_B : query.lora_Bs.1
            orig_dist += torch.norm(t1[n1]-t2[n2])
            orig_dist += torch.norm(t1[n1_B]-t2[n2_B])
            # P1 = torch.eye(p2.shape[0]).to(p2.device)
            # add noise to P1
            # P1 += torch.randn_like(P1) * 0.1
            # P1.requires_grad = True
            A_i = torch.zeros(p2.shape[0], p2.shape[0]).to(p2.device)
            # add noise to P2
            # P2 += torch.randn_like(P2) * 0.1
            A_i.requires_grad = True

            # to_optimize[n1] = {'P': P1}
            to_optimize[n2] = {'A_i': A_i}
    # Function to compute the matrix exponential
    def matrix_exponential(A):
        return torch.matrix_exp(A)
    optimizer = torch.optim.Adam([to_optimize[n]['A_i'] for n in to_optimize], lr=lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    loss_history = []
    for i in tqdm(range(num_training_steps)):
        optimizer.zero_grad()
        loss = 0
        A_reg_loss = 0
        B_reg_loss = 0
        sim_loss = 0
        for n in to_optimize:
            if n.endswith(f'.{moving_idx}'):
                # n1 = n
                # n2 = '.'.join(n.split('.')[:-1]) + f'.{moving_idx}'
                n1 = '.'.join(n.split('.')[:-1]) + f'.{ref_idx}'
                n2 = n

                n1_B = n1.replace('lora_As', 'lora_Bs')
                n2_B = n2.replace('lora_As', 'lora_Bs')
                # P1 = to_optimize[n1]['P']
                # BA => B(P@P^(-1))A
                B1 = t1[n1_B]
                B2 = t2[n2_B]
                A1 = t1[n1]
                A2 = t2[n2]

                A_i = to_optimize[n2]['A_i']
                P_i = matrix_exponential(A_i)

                P_iA2 = P_i @ A2

                norm_diff = ((P_iA2.norm(dim=1) - A2.norm(dim=1))**2).sum()
                reg_term = 0.1 * norm_diff
                loss += reg_term
                A_reg_loss += reg_term

                P_i_inv = torch.inverse(P_i)
                B2P_i_inv = B2 @ P_i_inv

                norm_diff = ((B2P_i_inv.norm(dim=0) - B2.norm(dim=0))**2).sum()
                reg_term = 0.1 * norm_diff
                loss += reg_term
                B_reg_loss += reg_term

                P_iA2_norm = P_iA2 / (P_iA2.norm(dim=1, keepdim=True) + 1e-8)
                A1_norm = A1 / (A1.norm(dim=1, keepdim=True) + 1e-8)

                cos_sim_matrix = P_iA2_norm @ A1_norm.T

                loss += (cos_sim_matrix ** 2).sum()
                sim_loss += (cos_sim_matrix ** 2).sum()
        if i % 10 == 0:
            print('loss : ', loss.item())
            print('A_reg_loss : ', A_reg_loss.item())
            print('B_reg_loss : ', B_reg_loss.item())
            print('sim_loss : ', sim_loss.item())
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    print('total distance : ', orig_dist)

    for n in t2:
        if 'lora_As' in n and n.endswith(f'.{moving_idx}'):
            n2 = n
            n1 = '.'.join(n.split('.')[:-1]) + f'.{ref_idx}'
            n1_B = n1.replace('lora_As', 'lora_Bs')
            n2_B = n.replace('lora_As', 'lora_Bs')
            # P1 = to_optimize[n1]['P']
            A_i = to_optimize[n2]['A_i']
            P_i = matrix_exponential(A_i)
            t1[n2] = P_i @ t2[n2]
            t1[n2_B] = t2[n2_B] @ torch.inverse(P_i)

            # t1[n1] = P1 @ t1[n1]
            # t1[n1_B] = t1[n1_B] @ torch.inverse(P1)

            # t2[n1] = (t2[n2] + t1[n1]) / 2
            # t2[n2] = (t2[n2] + t1[n1]) / 2
            # t2[n1_B] = (t2[n2_B] + t1[n1_B]) / 2
            # t2[n2_B] = (t2[n2_B] + t1[n1_B]) / 2

    return t1, t2, loss_history

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
    

    # make directory f'./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8 if not exist
    if not os.path.exists(f'./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8'):
        os.makedirs(f'./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')
    model_tomerge1 = f'./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir}/'
    model_state1 = torch.load(os.path.join(model_tomerge1, 'model.pt'), map_location='cpu')
    if 'Llama' in args.architecture:
        for n in list(model_state1.keys()):
            model_state1[n[17:-7]+'s.'+str(args.fixed_model)] = model_state1.pop(n)
    os.system(f'cp -a {model_tomerge1} ./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')

    
    # copy directory of model_tomerge1
    # os.system(f'rsync -av --exclude="model.pt" ./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/camera_unsup_{splitted_arch} ./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')

    for i in range(6):
        if i == args.fixed_model:
            continue
        model_tomerge2 = f'./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{dirs[i]}/'
        model_state2 = torch.load(os.path.join(model_tomerge2, 'model.pt'), map_location='cpu')
        if 'Llama' in args.architecture:
            for n in list(model_state2.keys()):
                model_state2[n[17:-7]+'s.'+str(i)] = model_state2.pop(n)
        model_state1, merged_params2, loss_history = trainclose_lerp(model_state1, model_state2, lr=1e-2, num_warmup_steps=50, num_training_steps=500, ref_idx = args.fixed_model, moving_idx = i)

        # plot the loss history
        #save loss_hitory plot
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.savefig('loss_history_ortho2.png')

        torch.save(model_state1, f'./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir}/model.pt')
    
    os.system(f'mv ./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir} ./New_lora_transform_AZeroCossim{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/camera_unsup_{splitted_arch}')




            






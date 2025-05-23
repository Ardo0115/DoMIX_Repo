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
            P2 = torch.eye(p2.shape[0]).to(p2.device)
            # add noise to P2
            # P2 += torch.randn_like(P2) * 0.1
            P2.requires_grad = True

            # to_optimize[n1] = {'P': P1}
            to_optimize[n2] = {'P': P2}

    optimizer = torch.optim.Adam([to_optimize[n]['P'] for n in to_optimize], lr=lr)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=1)
    loss_history = []
    I = torch.eye(8).cuda()
    for i in tqdm(range(num_training_steps)):
        optimizer.zero_grad()
        loss = 0
        for n in to_optimize:
            if n.endswith(f'.{moving_idx}'):
                # n1 = n
                # n2 = '.'.join(n.split('.')[:-1]) + f'.{moving_idx}'
                n1 = '.'.join(n.split('.')[:-1]) + f'.{ref_idx}'
                n2 = n

                n1_B = n1.replace('lora_As', 'lora_Bs')
                n2_B = n2.replace('lora_As', 'lora_Bs')
                # P1 = to_optimize[n1]['P']
                P2 = to_optimize[n2]['P']
                # BA => B(P@P^(-1))A
                B1 = t1[n1_B]
                B2 = t2[n2_B]
                A1 = t1[n1]
                A2 = t2[n2]

                loss += torch.norm(B1.t() @ B2 @ torch.inverse(P2)) **2
                loss += torch.norm((A1) @ (P2@ A2).t())**2
                # loss += torch.norm((P2@A2) @ (P2@A2).t() - I)
                # loss += torch.norm(torch.inverse(P2))**2

                # loss += torch.norm(B1 - B2 @ P2)**2
                # loss += torch.norm(A1 - torch.inverse(P2) @ A2 )**2
                # print all devices
                # print(B1.device, P1.device, B2.device, P2.device, A1.device, A2.device, I.device)
                # loss += torch.norm(((B1 @P1).t() @ B2 @ P2 - I))
                # loss += torch.norm((torch.inverse(P1) @ A1) @ (torch.inverse(P2) @ A2).t() - I)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    print('total distance : ', orig_dist)
    after_dist = 0
    with torch.no_grad():
        for n in to_optimize:
            if n.endswith(f'.{ref_idx}'):
                n1 = n
                n2 = '.'.join(n.split('.')[:-1]) + f'.{moving_idx}'
                n1_B = n1.replace('lora_As', 'lora_Bs')
                n2_B = n2.replace('lora_As', 'lora_Bs')
                # P1 = to_optimize[n1]['P']
                P2 = to_optimize[n2]['P']
                B1 = t1[n1_B]
                B2 = t2[n2_B]
                A1 = t1[n1]
                A2 = t2[n2]
                after_dist += torch.norm(B1 - B2 @ P2)
                after_dist += torch.norm(A1 - torch.inverse(P2) @ A2 )
    print('total distance after training : ', after_dist)

    for n in t2:
        if 'lora_As' in n and n.endswith(f'.{moving_idx}'):
            n2 = n
            n1 = '.'.join(n.split('.')[:-1]) + f'.{ref_idx}'
            n1_B = n1.replace('lora_As', 'lora_Bs')
            n2_B = n.replace('lora_As', 'lora_Bs')
            # P1 = to_optimize[n1]['P']
            P2 = to_optimize[n2]['P']
            t2[n2] = P2 @ t2[n2]
            t2[n2_B] = t2[n2_B] @ torch.inverse(P2)
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
    # parser.add_argument('--to_save_directory', type=str, required=True )

    args = parser.parse_args()

    dirs = ['restaurant_unsup_roberta', 'acl_unsup_roberta', 'ai_unsup_roberta', 'phone_unsup_roberta', 'pubmed_unsup_roberta', 'camera_unsup_roberta']
    ref_dir = dirs[args.fixed_model]
    
    model_tomerge1 = f'./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{ref_dir}/'
    model_state1 = torch.load(os.path.join(model_tomerge1, 'model.pt'), map_location='cpu')

    # make directory f'./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8 if not exist
    if not os.path.exists(f'./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8'):
        os.makedirs(f'./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')
    
    # copy directory of model_tomerge1
    os.system(f'cp -a {model_tomerge1} ./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')
    os.system(f'cp -a ./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/camera_unsup_roberta ./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8')
    for i in range(6):
        if i == args.fixed_model:
            continue
        model_tomerge2 = f'./seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{dirs[i]}/'
        model_state2 = torch.load(os.path.join(model_tomerge2, 'model.pt'), map_location='cpu')
        merged_params1, merged_params2, loss_history = trainclose_lerp(model_state1, model_state2, lr=1e-2, num_warmup_steps=100, num_training_steps=1000, ref_idx = args.fixed_model, moving_idx = i)

        # plot the loss history
        #save loss_hitory plot
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.savefig('loss_history_ortho2.png')

        # make directory if not exist
        if not os.path.exists(f'./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{dirs[i]}'):
            os.makedirs(f'./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{dirs[i]}')
        torch.save(merged_params2, f'./lora_transform_ABZero{ref_dir[:3].upper()}/seq0/Nonesamples/lora_pbatch16_ngpu4_lr5e-04_r8/{dirs[i]}/model.pt')




            






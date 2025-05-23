import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import transformers

ref_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/lora_r=8'
moving_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/math_lora_r=8'
# ref_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/math_lora_r=8'
# moving_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/lora_r=8'
# swap ref moving
# ref_dir, moving_dir = moving_dir, ref_dir
ref_lora = torch.load(os.path.join(ref_dir, 'adapter_model.bin'), map_location='cpu')
moving_lora = torch.load(os.path.join(moving_dir, 'adapter_model.bin'), map_location='cpu')

# Set device (CPU or GPU)
device = torch.device('cuda')  # Change to 'cuda' if using a GPU

torch.manual_seed(0)  # For reproducibility

# Assuming moving_lora and ref_lora are dictionaries (state_dicts) containing the model's weights
# You should have already loaded them before this code

# Get all keys ending with 'lora_B.weight'
lora_B_keys = [key for key in moving_lora.keys() if key.endswith('lora_B.weight')]

# Dictionaries to store variables for each layer
A_dict = {}       # Stores A_i matrices
B1_dict = {}      # Stores B1 matrices
A1_dict = {}      # Stores A1 matrices
B_ref_dict = {}   # Stores B_ref matrices
A_ref_dict = {}   # Stores A_ref matrices
r_dict = {}       # Stores r_i values (number of columns)

# Step 1: Load data for all layers and initialize A_i
for key in lora_B_keys:
    # Get B1 and B_ref for the current layer
    B1 = moving_lora[key].to(device)
    A1 = moving_lora[key.replace('lora_B', 'lora_A')].to(device)
    B_ref = ref_lora[key].to(device)
    A_ref = ref_lora[key.replace('lora_B', 'lora_A')].to(device)
    
    # Ensure B1 and B_ref have the same shape
    if B1.shape != B_ref.shape:
        print(f"Shapes of B1 {B1.shape} and B_ref {B_ref.shape} do not match for layer {key}. Skipping.")
        continue
    
    n, r = B1.shape  # Assuming B1 is of shape (n, r)
    r_dict[key] = r
    B1_dict[key] = B1
    A1_dict[key.replace('lora_B', 'lora_A')] = A1
    B_ref_dict[key] = B_ref
    A_ref_dict[key.replace('lora_B', 'lora_A')] = A_ref
    
    # Initialize A_i for this layer
    # A_i = torch.randn(r, r, device=device, requires_grad=True)
    A_i = torch.zeros(r, r, device=device, requires_grad=True)
    A_dict[key] = A_i

# Collect all A_i parameters for the optimizer
A_parameters = [A_dict[key] for key in A_dict.keys()]

# Function to compute the matrix exponential
def matrix_exponential(A):
    return torch.matrix_exp(A)

# Function to compute P_i as the exponential of a skew-symmetric matrix
def compute_P_orthogonal(A):
    # Ensure skew-symmetry: S = A - A^T
    S = A - A.T
    # Compute the matrix exponential
    P = torch.matrix_exp(S)
    return P

# Function to compute the total objective function
def total_objective_function(A_dict, B1_dict, A1_dict, B_ref_dict, A_ref_dict):
    total_loss = 0
    reg_loss_A = 0
    reg_loss_B = 0
    cossim_loss = 0
    for key in A_dict.keys():
        A_i = A_dict[key]
        B1 = B1_dict[key]
        A1 = A1_dict[key.replace('lora_B', 'lora_A')]
        B_ref = B_ref_dict[key]
        A_ref = A_ref_dict[key.replace('lora_B', 'lora_A')]
        P_i = matrix_exponential(A_i)
        BP_i = B1 @ P_i

        # # Normalize BP_i and B_ref to unit vectors column-wise
        # BP_i_norm = BP_i / (BP_i.norm(dim=0, keepdim=True) + 1e-8)  # Adding epsilon for numerical stability
        # B_ref_norm = B_ref / (B_ref.norm(dim=0, keepdim=True) + 1e-8)
        
        # # Compute the cosine similarity for all columns at once
        # cos_sim = (BP_i_norm * B_ref_norm).sum(dim=0)  # Element-wise multiplication and sum over columns
        
        # # Sum over all column similarities to minimize their total
        # loss_i = (cos_sim**2).sum()

        # # Scale BP_i to match the norms of B1
        # B1_norms = B1.norm(dim=0, keepdim=True)
        # BP_i_norms = BP_i.norm(dim=0, keepdim=True) + 1e-8
        # BP_i = BP_i * (B1_norms / BP_i_norms)  # Scale to have the same norms as B1


        # Compute the norm difference as a regularization term
        norm_diff = ((BP_i.norm(dim=0) - B1.norm(dim=0)) ** 2).sum()
        reg_term_B = 0.1 * norm_diff  # Regularization coefficient (adjust as needed)
        reg_loss_B += reg_term_B


        P_i_inv = torch.inverse(P_i)
        P_i_inv_A1 = P_i_inv @ A1
        norm_diff = ((P_i_inv_A1.norm(dim=1) - A1.norm(dim=1)) ** 2).sum()
        reg_term_A = 0.1 * norm_diff  # Regularization coefficient (adjust as needed)
        reg_loss_A += reg_term_A

        # Normalize BP_i and B_ref to unit vectors column-wise
        BP_i_norm = BP_i / (BP_i.norm(dim=0, keepdim=True) + 1e-8)
        B_ref_norm = B_ref / (B_ref.norm(dim=0, keepdim=True) + 1e-8)
        
        # Compute the cosine similarities between all column pairs
        cos_sim_matrix = BP_i_norm.T @ B_ref_norm  # Shape will be (n, n) if both have n columns
        
        # Square the cosine similarity matrix and sum all elements to get the total loss for this key
        cossim_loss += (cos_sim_matrix ** 2).sum()


        # loss_i = torch.norm(BP_i.T @ B_ref, p='fro')**2
    total_loss = cossim_loss + reg_loss_B + reg_loss_A
    return total_loss, reg_loss_B, reg_loss_A, cossim_loss
# Lists to store loss values for plotting
loss_history = []


# Step 3: Optimization loop with tqdm progress bar
num_epochs = 500
num_warmup_steps = 50
tolerance = 1e-6
prev_loss = float('inf')

# Step 2: Set up the optimizer
optimizer = optim.Adam(A_parameters, lr=0.01)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_epochs)




print("Starting optimization...")

# Step 4: After optimization, retrieve P_i for each layer and update moving_lora
P_matrices = {}
import math
for key in A_dict.keys():
    moving_lora[key] = torch.nn.init.normal_(moving_lora[key], mean=0.0, std=0.001)
    key_A = key.replace('lora_B.weight', 'lora_A.weight')
    moving_lora[key_A] = torch.nn.init.normal_(moving_lora[key_A], mean=0.0, std=0.001)

# Step 5: Save the transformed state_dict (moving_lora)
save_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/Rank8_Normal_Initialized'
os.makedirs(save_dir, exist_ok=True)
torch.save(moving_lora, os.path.join(save_dir, 'adapter_model.bin'))
print("\nTransformed moving_lora state_dict saved as 'transformed_moving_lora.pth'.")

# Step 6: Plot and save the optimization loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Optimization Loss Curve')
plt.grid(True)
plt.savefig('optimization_loss_curve.png')
plt.show()
print("Optimization loss curve saved as 'optimization_loss_curve.png'.")

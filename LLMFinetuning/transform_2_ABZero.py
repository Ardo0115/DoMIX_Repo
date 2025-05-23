import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# ref_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/lora_r=8'
# moving_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/math_lora_r=8'
ref_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/math_lora_r=8'
moving_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/lora_r=8'
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
B_ref_dict = {}   # Stores B_ref matrices
r_dict = {}       # Stores r_i values (number of columns)
A1_dict = {}      # Stores A1 matrices
A_ref_dict = {}   # Stores A_ref matrices

# Step 1: Load data for all layers and initialize A_i
for key in lora_B_keys:
    # Get B1 and B_ref for the current layer
    B1 = moving_lora[key].to(device)
    B_ref = ref_lora[key].to(device)
    key_A = key.replace('lora_B.weight', 'lora_A.weight')
    A1 = moving_lora[key_A].to(device)
    A_ref = ref_lora[key_A].to(device)
    
    # Ensure B1 and B_ref have the same shape
    if B1.shape != B_ref.shape:
        print(f"Shapes of B1 {B1.shape} and B_ref {B_ref.shape} do not match for layer {key}. Skipping.")
        continue
    
    n, r = B1.shape  # Assuming B1 is of shape (n, r)
    r_dict[key] = r
    B1_dict[key] = B1
    B_ref_dict[key] = B_ref
    A1_dict[key_A] = A1
    A_ref_dict[key_A] = A_ref
    
    # Initialize A_i for this layer
    A_i = torch.randn(r, r, device=device, requires_grad=True)
    A_dict[key] = A_i

# Collect all A_i parameters for the optimizer
A_parameters = [A_dict[key] for key in A_dict.keys()]

# Function to compute the matrix exponential
def matrix_exponential(A):
    return torch.matrix_exp(A)

# Function to compute the total objective function
def total_objective_function(A_dict, B1_dict, B_ref_dict, A1_dict, A_ref_dict):
    total_loss = 0
    for key in A_dict.keys():
        A_i = A_dict[key]
        B1 = B1_dict[key]
        B_ref = B_ref_dict[key]
        key_A = key.replace('lora_B.weight', 'lora_A.weight')
        A1 = A1_dict[key_A]
        A_ref = A_ref_dict[key_A]
        P_i = matrix_exponential(A_i)
        BP_i = B1 @ P_i
        P_i_inv = torch.inverse(P_i)
        P_i_inv_A1 = P_i_inv @ A1
        loss_i = torch.norm(BP_i.T @ B_ref, p='fro')**2
        loss_i += torch.norm(A_ref @ P_i_inv_A1.T, p='fro')**2
        total_loss += loss_i
    return total_loss

# Lists to store loss values for plotting
loss_history = []

# Step 2: Set up the optimizer
optimizer = optim.Adam(A_parameters, lr=0.01)

# Step 3: Optimization loop with tqdm progress bar
num_epochs = 5000
tolerance = 1e-6
prev_loss = float('inf')

print("Starting optimization...")
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    optimizer.zero_grad()
    loss = total_objective_function(A_dict, B1_dict, B_ref_dict, A1_dict, A_ref_dict)
    loss.backward()
    optimizer.step()
    
    # Store loss for plotting
    loss_history.append(loss.item())
    
    # Check for convergence
    if abs(prev_loss - loss.item()) < tolerance:
        print(f"\nConverged at epoch {epoch}")
        break
    prev_loss = loss.item()

# Step 4: After optimization, retrieve P_i for each layer and update moving_lora
P_matrices = {}
for key in A_dict.keys():
    A_i = A_dict[key]
    P_i = matrix_exponential(A_i).detach()
    P_matrices[key] = P_i

    # Verify that P_i is invertible
    det_P = torch.det(P_i)
    if det_P.abs() > 1e-6:
        print(f"\nLayer: {key}")
        print(f"P_i is invertible with determinant {det_P.item():.6f}")
    else:
        print(f"\nLayer: {key}")
        print("P_i is not invertible or close to singular.")
    
    # Compute the objective function value with P_i
    B1 = B1_dict[key]
    B_ref = B_ref_dict[key]
    key_A = key.replace('lora_B.weight', 'lora_A.weight')
    A1 = A1_dict[key_A]
    A_ref = A_ref_dict[key_A]
    objective_optimal = torch.norm((B1 @ P_i).T @ B_ref, 'fro')**2
    objective_optimal += torch.norm(A_ref @ (torch.inverse(P_i) @ A1).T, 'fro')**2
    print(f"Objective function value with optimal P_i: {objective_optimal.item():.6f}")
    
    # Compare with the closed-form solution
    # Compute C_i = B1.T @ B_ref
    C_i = B1.T @ B_ref
    # Perform SVD on C_i
    U_i, S_i, Vh_i = torch.linalg.svd(C_i, full_matrices=False)
    V_i = Vh_i.T
    # Compute P_closed = U_i @ V_i.T
    P_closed = U_i @ V_i.T
    # Compute the objective function value with P_closed
    objective_closed = torch.norm((B1 @ P_closed).T @ B_ref, 'fro')**2
    objective_closed += torch.norm(A_ref @ (torch.inverse(P_closed) @ A1).T, 'fro')**2
    print(f"Objective function value with closed-form P_i: {objective_closed.item():.6f}")
    
    # Update moving_lora weights using the optimized P_i
    # Apply the transformation to B1: B1_new = B1 @ P_i
    moving_lora[key] = (B1 @ P_i).detach().cpu()
    key_A = key.replace('lora_B.weight', 'lora_A.weight')
    moving_lora[key_A] = (torch.inverse(P_i) @ moving_lora[key_A].to(device)).detach().cpu()

# Step 5: Save the transformed state_dict (moving_lora)
save_dir = '/home/ardo0115/workspace/Subspace-Tuning/CR_MR/finetune/ABzero_transformed_from-lora-to-Numerical'
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

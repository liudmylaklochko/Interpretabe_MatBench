#%%
import torch, json
import SAE.sparce_autoencoder as SAE
torch.backends.cudnn.enabled=False
import pandas as pd
import utils_f as  u 
import numpy as np 
import json 

def check_dead_neurons(decoded_final, activation_data, k=1.0, low_var_threshold=0.10):
    """
    Compute indices of neurons that are dead (low variance) or poorly reconstructed (high MSE)
    """
    activation_np = activation_data.detach().cpu().numpy()
    decoded_np = decoded_final.detach().cpu().numpy()

    mse_per_neuron = np.mean((activation_np - decoded_np)**2, axis=0)
    std_per_neuron = np.std(activation_np, axis=0)

    high_error_threshold = np.mean(mse_per_neuron) + k * np.std(mse_per_neuron)
    dead_indices = np.where((mse_per_neuron > high_error_threshold) | (std_per_neuron < low_var_threshold))[0]

    print(f"  → {len(dead_indices)} dead neurons detected")
    return dead_indices

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu()
    return hook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name='CGCNN'

data = torch.load(f'../activations_{model_name}/activations.pt')
config = json.load(open(f"configurations/config_{model_name}.json"))
possible_layers = pd.read_csv(f'../activations_{model_name}/non_empty_layers.txt')

if 'activations' in data:
    batch = data['activations']
else:
    batch = data 

if 'mpd_ids' in data:
    batch_mp_ids = data['mpd_ids']
elif 'mp_ids' in data:
    batch_mp_ids = data['mp_ids']


encoded_outputs = {}
SAEs = {}
SAEev = {}
optimizers = {}

idx_layer = {} 
filtered_acts = {} 

for layer in possible_layers.layers:
    if model_name != 'ALIGNN':
        x = torch.stack(batch[layer], dim=0).to(device)
    else:
        x = (batch[layer]).to(device)

    SAEs[layer] = torch.load(f"{config['saedir']}/{layer}.pkl").to(device)
    SAEs[layer].eval()   
   
    with torch.no_grad():
        decoded, encoded = SAEs[layer](x)

    dead_idx = check_dead_neurons(decoded, x)
    idx_layer[layer] = dead_idx
    x_filtered = encoded.detach().cpu().numpy()

    if len(dead_idx) > 0:
        x_filtered = np.delete(x_filtered, dead_idx, axis=1)
        print(f"  filtered {len(dead_idx)} neurons, remaining {x_filtered.shape[1]}")
    encoded_outputs[layer] = torch.tensor(x_filtered)

    #with torch.no_grad():
    #    _ = SAEs[layer](x)
    #encoded_outputs[layer] = activation[layer]    


torch.save(idx_layer,f"inactive_idx_{model_name}.pt")


for layer in possible_layers.layers:
    if model_name != 'ALIGNN':
        x = torch.stack(batch[layer], dim=0).to(device)
    else:
        x = (batch[layer]).to(device)   
    filtered_acts[layer] = torch.from_numpy(np.delete(x.detach().cpu().numpy(), idx_layer[layer], axis=1))

torch.save({'activations': filtered_acts, 'mp_ids': batch_mp_ids}, f'../activations_{model_name}/filtered_base_activations_{model_name}.pt')
torch.save({'activations': encoded_outputs, 'mp_ids': batch_mp_ids}, f'activations_SAE_{model_name}.pt')



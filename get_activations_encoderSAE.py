#%%
import torch, json
import SAE.sparce_autoencoder as SAE
torch.backends.cudnn.enabled=False
import pandas as pd
import utils_f as  u 
import numpy as np 
import json 

torch.backends.cudnn.enabled = False
torch.set_grad_enabled(False)

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
device = 'cpu'

for layer in possible_layers.layers:
    if model_name != 'ALIGNN':
        x = torch.stack(batch[layer], dim=0).to(device)
    else:
        x = (batch[layer]).to(device)

    SAEs[layer] = torch.load(f"{config['saedir']}/{layer}.pkl").to(device)
    SAEs[layer].eval()   
   
    with torch.no_grad():
        _, encoded = SAEs[layer](x)

    encoded_outputs[layer] = encoded.cpu().detach().numpy()
  

torch.save({'activations': encoded_outputs, 'mp_ids': batch_mp_ids}, f'activations_SAE_{model_name}.pt')



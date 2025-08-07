#%%
import torch, json
import SAE.sparce_autoencoder as SAE
torch.backends.cudnn.enabled=False
import pandas as pd
import utils_f as  u 


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu()
    return hook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load('../activations_MEGNet/activations.pt')
config = json.load(open("configurations/config_MegNet.json"))
possible_layers = pd.read_csv('../activations_MEGNet/non_empty_layers.txt')


batch = data['activations']
batch_mp_ids = data['mp_ids']

encoded_outputs = {}

SAEs = {}
SAEev = {}
optimizers = {}

for layer in possible_layers.layers:
    activation = {}
    x = torch.stack(batch[layer], dim=0).to(device)

    SAEs[layer] = torch.load(f"{config['saedir']}/{layer}.pkl").to(device)
    SAEs[layer].eval()   
    SAEs[layer].encoder.register_forward_hook(get_activation(layer))
    
    with torch.no_grad():
        _ = SAEs[layer](x)
    
    encoded_outputs[layer] = activation[layer]    


torch.save({'activations': batch, 'mp_ids': batch_mp_ids}, 'activations_SAE.pt')



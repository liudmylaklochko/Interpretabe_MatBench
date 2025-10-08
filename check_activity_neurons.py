#%%
import torch, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SAE.sparce_autoencoder as SAE
torch.backends.cudnn.enabled=False
import pandas as pd
import utils_f as  u 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")


def check_dead_neurons(self, decoded, activation_data):
    """
    I check with MSE diff which idx of neurons are dead
    """
    mse_per_neuron = np.mean((activation_data.cpu().numpy() - decoded_final.detach().cpu().numpy()) ** 2, axis=0)
    std_per_neuron = np.std(activation_data.cpu().numpy(), axis=0)
    k=1.0
    
    mean_err = np.mean(mse_per_neuron)
    std_err = np.std(mse_per_neuron)
    
    high_error_threshold = mean_err + k * std_err
    low_var_threshold = 0.10
    
    bad_reconstruction = mse_per_neuron > high_error_threshold
    low_variability = std_per_neuron < low_var_threshold
    dead_indices = np.where(low_variability | bad_reconstruction)[0]
    print(f"Dead neurons: {dead_indices}")


def visualize_diagnostics(decoded):
    """
    plot of the heatmap of decoded activatios
    Y xias: number of neuron
    X axis : sampe, i.e. material 
    """
    decoded = decoded.detach().cpu().numpy()
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(decoded.T, 
                cmap='viridis', ax=ax2, cbar_kws={'label': 'Decoded Activation'})
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Neuron')
    plt.show()

#======= 



model_name='CGCNN'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"): print("USING GPU")


config = json.load(open(f"configurations/config_{model_name}.json"))

torchseed = config['seed'] 
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)


print("********  Loading activations  ********")

data = torch.load(f'../activations_{model_name}/activations.pt')
batch = data#['activations']
possible_layers = pd.read_csv(f'../activations_{model_name}/non_empty_layers.txt')

SAEs = {}
SAEev = {}
optimizers = {}
for layer in possible_layers.layers:
    SAEs[layer] = torch.load(f"{config['saedir']}/{layer}.pkl")    


layer = possible_layers.layers[0]
decoded_final, encoded_final = SAEs[layer](torch.stack(batch[layer], dim=0).to(device))
u.check_neuron(torch.stack(batch[layer], dim=0).to(device).cpu(),decoded_final.detach().cpu(),neuron_index=29)


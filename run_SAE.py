#%%
import torch, json
import SAE.sparce_autoencoder as SAE
torch.backends.cudnn.enabled=False
import pandas as pd
import utils_f as  u 

def trainSAE(SAEs, optimizers, layer, acts, SAEev):
    if layer not in SAEs:
        factor = config["basefactor"] 
        while acts.size()[1]*acts.size()[1]*factor*4 > config["maxsize"]:
            factor -= 1
        if factor < 1: 
            SAEs[layer] = None   
            print(f"{layer} too large, skipping")
        else:    
            encoding_dim = int(acts.size()[1] * factor)
            #print(f"Encoding dim for {layer} = {encoding_dim} (factor={factor})")
            SAEs[layer] = SAE.SparseAutoencoder(acts.size()[1], 
                                                encoding_dim, 
                                                beta=config["beta"], 
                                                rho=config["rho"]).to(device) 
            SAEev[layer] = {"rec": 0, "sparse": 0, "loss": 0, "min": None}
            optimizers[layer] = torch.optim.Adam(SAEs[layer].parameters(), 
                                                 lr=config["learningrate"])
            SAEs[layer].train()
    if SAEs[layer] is None: return
    optimizers[layer].zero_grad()
    decoded, encoded = SAEs[layer](acts)
    total_loss, recon_loss_val, sparsity_val = SAEs[layer].compute_loss(acts, decoded, encoded)  
    total_loss.backward()
    optimizers[layer].step()
    SAEev[layer]["rec"] += recon_loss_val
    SAEev[layer]["sparse"] += sparsity_val
    SAEev[layer]["loss"] += total_loss




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"): print("USING GPU")

config = json.load(open("configurations/config_MegNet.json"))

torchseed = config['seed'] 
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)


print("********  Loading activations  ********")

#batch = torch.load('../activations_MEGNet/activations.pt')
data = torch.load('../activations_MEGNet/activations.pt')
batch = data['activations']
possible_layers = pd.read_csv('../activations_MEGNet/non_empty_layers.txt')



print("********  Training SAE  ********")


SAEs = {}
SAEev = {}
optimizers = {}


tl_ep = pd.DataFrame(columns=possible_layers.layers)
tl_ep.index.rename('epoch', inplace=True)

rc_ep = pd.DataFrame(columns=possible_layers.layers)
rc_ep.index.rename('epoch', inplace=True)

sp_ep = pd.DataFrame(columns=possible_layers.layers)
sp_ep.index.rename('epoch', inplace=True)


for ep in range(0, config["nepochs_sae"]+1):
    print("Epoch: ", ep)
    count = 0
    sev = 0
    tl = {} 
    rc = {} 
    sp = {} 

    for layer in possible_layers.layers:
        trainSAE(SAEs, optimizers, layer, torch.stack(batch[layer], dim=0).to(device), SAEev)
        count = len (batch[layer])

        tl[layer] = SAEev[layer]['loss'].detach().cpu().item()
        rc[layer] = SAEev[layer]['rec']
        sp[layer] = SAEev[layer]['sparse']

    tl_ep.loc[len(tl_ep)] = tl
    rc_ep.loc[len(rc_ep)] = rc
    sp_ep.loc[len(sp_ep)] = sp

    for layer in SAEev:
        torch.save(SAEs[layer], f"{config['saedir']}/{layer}.pkl")
        SAEev[layer]["rec"] /= count
        SAEev[layer]["sparse"] /= count
        SAEev[layer]["loss"] /= count
        print(f"   {layer}:: rec={SAEev[layer]['rec']}, sparse={SAEev[layer]['sparse']}, loss={SAEev[layer]['loss']}")



print("********  Visualize neuron activity of SAE  ********")


# load trained SAE
""" config = json.load(open("configurations/config_MegNet.json"))
SAEs = {}
SAEev = {}
optimizers = {}
possible_layers = pd.read_csv('../activations_MEGNet/non_empty_layers.txt')
for layer in possible_layers.layers:
    SAEs[layer] = torch.load(f"{config['saedir']}/{layer}.pkl")
data = torch.load('../activations_MEGNet/activations.pt')
batch = data['activations']
mpd_ids = data['mpd_ids']  
 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


layer = possible_layers.layers[128]
decoded_final, encoded_final = SAEs[layer](torch.stack(batch[layer], dim=0).to(device))
u.check_neuron(torch.stack(batch[layer], dim=0).to(device).cpu(),decoded_final.detach().cpu(),neuron_index=5)
u.visualize_neuron_activity_all(encoded_final.detach().cpu(), display_count=12, row_length=4)
u.plot_losses(tl_ep[layer],rc_ep[layer], sp_ep[layer])


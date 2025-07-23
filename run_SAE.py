#%%
import torch, sys, json
import SAE.sparce_autoencoder as SAE
torch.backends.cudnn.enabled=False
import pandas as pd

def trainSAE(SAEs, optimizers, layer, acts, SAEevs):
    if layer not in SAEs:
        factor = config["basefactor"] 
        while acts.size()[1]*acts.size()[1]*factor*4 > config["maxsize"]:
            factor -= 1
        if factor < 1: 
            SAEs[layer] = None   
            print(f"{layer} too large, skipping")
        else:    
            encoding_dim = int(acts.size()[1] * factor)
            print(f"Encoding dim for {layer} = {encoding_dim} (factor={factor})")
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

batch = torch.load('../activations_MEGNet/activations.pt')
possible_layers = pd.read_csv('../activations_MEGNet/non_empty_layers.txt')

#layer = possible_layers.layers[1]
#print ("Train will be done for :",layer)

print("********  Training SAE  ********")


SAEs = {}
SAEev = {}
optimizers = {}

for ep in range(1, config["nepochs_sae"]+1):
    count = 0
    sev = 0

    for layer in possible_layers.layers:
        trainSAE(SAEs, optimizers, layer, torch.stack(batch[layer], dim=0).to(device), SAEev)
        count = len (batch[layer])

    for layer in SAEev:
        torch.save(SAEs[layer], f"{config['saedir']}/{layer}.pkl")
        SAEev[layer]["rec"] /= count
        SAEev[layer]["sparse"] /= count
        SAEev[layer]["loss"] /= count
        print(f"   {layer}:: rec={SAEev[layer]['rec']}, sparse={SAEev[layer]['sparse']}, loss={SAEev[layer]['loss']}")

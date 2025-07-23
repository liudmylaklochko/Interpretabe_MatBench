
#%%
import torch, sys, json
import utils_f as u
import pickle
torch.backends.cudnn.enabled=False
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"): print("USING GPU")

config = json.load(open("configurations/config_MegNet.json"))

torchseed = config['seed'] 
torch.manual_seed(torchseed)
torch.cuda.manual_seed(torchseed)


print("********  Loading dataset  ********")

batch = torch.load('../activations_MEGNet/activations.pt')
possible_layers = pd.read_csv('../activations_MEGNet/non_empty_layers.txt')

layer = possible_layers.layers[1]

print ("Train will be done for :",layer)

print("********  Training SAE  ********")


SAEs = {}
SAEev = {}
optimizers = {}

for ep in range(1, config["nepochs_sae"]+1):
    count = 0
    sev = 0
    # for X,y in data_loader:
    for i, struct in enumerate(structures):
        u.activation = {}
        with torch.no_grad():
        pred = model.predict_structure(struct)        
        for layer in u.activation:
        acts = get_activations_megnet(u.activation[layer])
        if acts is None or len(acts) <= 1 : continue
        # print(layer, "::", acts.shape)
        # create a batch of batch_size
        if layer not in batch: batch[layer] = []
        else: 
            if len(batch[layer]) != 0 and len(acts) != len(batch[layer][0]):
                # print("Bad layer: ", layer)
                batch[layer][0] = []
            else: 
                batch[layer].append(acts)
        if len(batch[layer]) >= config["batchsize"]:
            #print("layer batch", layer) 
            # print(len(batch[layer]))
            # trainSOM(torch.stack(batch[layer], dim=0), layer, SOMs, mm, SOMevs)
            # print(".", end="")
            trainSAE(SAEs, optimizers, layer, torch.stack(batch[layer], dim=0), SAEev)
            batch[layer] = []
        count += 1
        if i%100==0: print(".", end="")
    print()        

    for layer in SAEev:
        torch.save(SAEs[layer], f"{config['saedir']}/{layer}.pkl")
        SAEev[layer]["rec"] /= count
        SAEev[layer]["sparse"] /= count
        SAEev[layer]["loss"] /= count
        print(f"   {layer}:: rec={SAEev[layer]['rec']}, sparse={SAEev[layer]['sparse']}, loss={SAEev[layer]['loss']}")

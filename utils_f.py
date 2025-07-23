

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
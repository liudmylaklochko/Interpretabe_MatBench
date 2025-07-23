#%%
import torch
import json, sys, os 
import utils_f   as u 
import importlib as imp
import numpy     as np
import pandas    as pd
from SPE.sparce_autoencoder import SPE_Integrate
from return_predictions import *
import pickle
import random


def unwrap_model(model):
    # Common patterns for wrapped models
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        return model.model
    return model

def run_case_by_module(module_name: str):
    try:
        module = imp.import_module(module_name)
        activations = module.run_acivations(calculate_act = False) # if we save them, if no: True
        return activations
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        return None

if __name__ == "__main__":

    if len(sys.argv) != 2:
            print("Usage: python run_individual_cond_exp.py <model name>")
            sys.exit(1)
            
    model_name=sys.argv[1]

    case_modules = [
        "create_activations_%s"%(model_name)
    ]

    config = json.load(open("configurations/ice/config_ice_%s.json"%(model_name)))

    

    torch.manual_seed(config['seed'])
    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"): print("USING GPU")
    
    for mod in case_modules:
        print(f"\n=== Running {mod} ===")
        res = run_case_by_module(mod)
        base_model = res[2]
        

    #list_layers = res[3]

    with open (config['saedir']+'/list_of_layers_trained_SAE.pkl', 'rb') as fp:
        list_layers = pickle.load(fp)
    
    print("*****  Select an index of a  layer from the possible list on which we trained SAE (\n *****")
    
    for idx, layer_name in enumerate(list_layers):
        print(f"id: {idx} for layer: {layer_name} ")
    

    selected_layer = list_layers[int(input("Enter the ID of the layer you want to select: "))]


    base_model = unwrap_model(base_model)
    spe_path = "SPE"
    sys.path.insert(0, spe_path)

    sparse_model_path = f"{config['saedir']}{selected_layer}.pkl" 
    spe_model =  u.load_model(sparse_model_path, device=device)

    base_model.eval()
    spe_model.eval()


    target_layer = u.get_module_by_name(base_model, selected_layer)
    target_layer_out_dim = u.get_first_linear_out_features(target_layer)

    while True:
        if spe_model.encoder[0].in_features == target_layer_out_dim:
            break
        print("Please, select another layer from the list above:")
        for i, layer_name in enumerate(list_layers):
            print(f"{i}: {layer_name}")
        try:
            selected_id = int(input("Enter the ID of the layer you want to select: "))
            selected_layer = list_layers[selected_id]
            sparse_model_path = f"{config['saedir']}{selected_layer}.pkl" 
            spe_model =  u.load_model(sparse_model_path, device=device)
            target_layer = u.get_module_by_name(base_model, selected_layer)
            target_layer_out_temp = u.get_first_linear_out_features(target_layer)
            if target_layer_out_temp == spe_model.encoder[0].in_features:
                break
            else:
                print(f"Selected layer's output features ({target_layer_out_temp}) do not match target ({target_layer_out_dim}). Try again.\n")
        except (IndexError, ValueError) as e:
            print("Invalid input. Please enter a valid layer ID.\n")



    print (f'********   Checking if the layer exists in {base_model.__class__.__name__} model   ********')

    if target_layer is None:
        print(f"Warning: Layer '{selected_layer}' not found in the base_model.")
        sys.exit()
    else:
        print ('********   Layer exists, starting perturbation   ********')    
            

    feature_index = config['feature_index']
    if target_layer_out_dim == None:
        target_layer_out_dim = target_layer_out_temp

    if feature_index > (target_layer_out_dim - 1):
        print ("Select the proper feature index! ")
        feature_index = int(input(f"id of a feature index (max = {target_layer_out_dim- 1}): "))

    perturb_range = (config['p_start'], config['p_stop'],config['p_step'])
    perturb_values = np.arange(*perturb_range)

    spe_injection  = SPE_Integrate(base_model=target_layer, spe_model=spe_model, feature_index=feature_index)
    u.set_module_by_name(base_model, selected_layer, spe_injection)

    print("********    Loading dataset    ********")

    data_loader = res[1] 

    print ('********   ICE calcualtion is starting   ********')

    function_name = f"predict_{model_name}"
    results = globals()[function_name](base_model, data_loader, perturb_values, spe_injection, device)
            
    print ('********   ICE calcualtion is done   ********')    
                 
    df = pd.DataFrame(results)
    df.index.name = 'batch_index'
    u.plot_ICE(df,perturb_values,feature_index, selected_layer,config)
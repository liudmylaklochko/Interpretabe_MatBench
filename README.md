# Overview 


This project applies interpretability techniques to neural networks trained on materials science tasks taken from MatBench. 
We extract activations from models trained on **energy formation** task and use Sparse Autoencoders to identify interpretable features in the learned representations.

# Supported models: 

## Crystal Graph Convolutional Neural Network (CGCNN)
 Graph-based architecture for crystal structure property prediction.
    - Suitable for periodic crystal structures
    **Activation repo:** activations_CGCNN
## ALIGNN (Atomistic Line Graph Neural Network)
Advanced graph neural network using both atom and bond graphs.
    - State-of-the-art performance on many materials benchmarks
    **Activation repo:** activations_ALIGNN
## MEGNet (MatErials Graph Network )   
The MatErials Graph Network (MEGNet) is an implementation of DeepMind's graph networks
Materials graph network for property prediction.
    - Flexible framework for various material representations
    **Activation repo:** activations_MEGNet
## CrabNet (Compositionally-Restricted Attention-Based Network )
Transformer-based model for materials property prediction from compositions.
    - Uses self-attention mechanisms on elemental compositions
    - Works directly from chemical formulas without structure data
    **Activation repo:** activations_CrabNet


# Datasets
- **Materials Project**


# Sparse Autoencoder Training:
- **Reconstruction + Sparsity:** KL-divergence penalty
- **Loss Function:** MSE reconstruction + KL-divergence sparsity penalty


# Citation 

If you use this code in your research, please cite: 
....

# Acknowledgments

- CGCNN   implementation from CGCNN   repository
- CrabNet implementation from CrabNet repository
- MEGNet  implementation from MEGNet  repository 
- ALIGNN  implementation from ALIGNN  repository 
- Sparse Autoencoder concepts from Anthropic's interpretability research
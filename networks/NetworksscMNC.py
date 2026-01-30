import numpy as np
import torch.nn as nn
import torch


class scEncoder(nn.Module):
    """
    Adopted from ConvNetworksPolyMNIST.py, itself adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(scEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(  # input shape (, input_dim)
            nn.Linear(self.input_dim, self.hidden_dim),  
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),  
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),  
            nn.ReLU(),
        )
         # latent representation
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return (
            self.mu(h),
            self.logvar(h),
        )

class scDecoder(nn.Module):
    """
    Adopted from ConvNetworksPolyMNIST.py, itself adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, output_dim, latent_dim, hidden_dim):
        super(scDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),  
            nn.ReLU(),
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(
            z.device
        )  # NOTE: consider learning scale param, too
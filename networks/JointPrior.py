import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogMat(nn.Module):
    def __init__(self, cfg):
        super(OrthogMat, self).__init__()
      
        # Parametrize the eigenvectors and eigenvalues
        self.latent_dim = cfg.model.laten_dim
        self.eigenvectorsU = nn.Parameter(torch.rand(self.latent_dim,
                                                     self.latent_dim ))  # Uniform initialization
        self.eigenvectorsV = nn.Parameter(torch.rand(self.latent_dim,
                                                     self.latent_dim ))  # Uniform initialization
        self.eigenvalues = nn.Parameter(torch.rand(self.latent_dim))  # Uniform initialization

    def forward(self, x):
        # Orthogonalize the eigenvectors (using QR decomposition)
        # U: upper triangular part, subtract transpose to create a skew-symmetric matrix
        U = torch.triu(self.eigenvectorsU, diagonal=1)
        A = U - U.T
        
        O_U = torch.matmul(torch.eye(self.latent_dim).to(x.device) + A, torch.linalg.inv(torch.eye(self.latent_dim).to(x.device) - A))
        # swap to uing solve rather than inv
        # V: similarly orthogonalize eigenvectorsV
        V = torch.triu(self.eigenvectorsV, diagonal=1)
        B = V - V.T
        O_V = torch.matmul(torch.eye(self.latent_dim).to(x.device) + B, torch.linalg.inv(torch.eye(self.latent_dim).to(x.device) - B))
        
        # Apply the orthogonal matrices to the input x
        return torch.matmul(O_U, x), torch.sigmoid(self.eigenvalues), O_V
        return 

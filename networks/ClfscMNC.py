# code obtained from asking chatgpt to "modify this[ClfImgPolyMNIST.py] to work for continuous data inputs rather than images", although had to change this quite a lot
import torch.nn as nn
import torch.nn.functional as F

# simple activation fn wrapper
def actvn(x):
    return F.leaky_relu(x, 0.2)

class BlockFC(nn.Module):
    """block for fully-connected (vector) data."""
    def __init__(self, fin, fout, is_bias=True):
        super().__init__()
        self.is_bias = is_bias
        self.fin = fin
        self.fout = fout
        self.fc0 = nn.Linear(self.fin, self.fout, bias=is_bias)
    
    def forward(self, x):
        dx = self.fc0(x)
        return actvn(dx)

class ClfscMNC_FC(nn.Module):
    """Classifier for continuous (vector) inputs."""
    def __init__(self, n_features, n_classes=6, hidden_dim=128, nblocks=3):
        super().__init__()
        layers = [BlockFC(n_features, hidden_dim)]
        for _ in range(nblocks - 1):
            layers.append(BlockFC(hidden_dim, hidden_dim))

        self.resnet = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, n_classes)
        self.encoder = nn.Sequential(  # input shape (, input_dim)
            nn.Linear(n_features, 2*n_features),  
            nn.BatchNorm1d(2*n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(2*n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.LeakyReLU(),
            nn.Dropout(0.6),
        )
        self.final_layer = nn.Linear(n_features, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.encoder(x)
        # out = self.fc_out(actvn(out))
        # out = self.fc_out(actvn(out))
        out = self.final_layer(out)
        out = self.softmax(out)
        return out

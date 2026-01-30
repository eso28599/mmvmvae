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

    def forward(self, x):
        out = self.resnet(x)
        out = self.fc_out(actvn(out))
        return out

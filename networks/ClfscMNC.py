import torch.nn as nn

class ClfscMNC_FC(nn.Module):
    """Classifier for continuous (vector) inputs."""
    def __init__(self, n_features, n_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 2*n_features),  
            nn.BatchNorm1d(2*n_features),
            nn.LeakyReLU(),
            nn.Linear(2*n_features, n_features),
            nn.BatchNorm1d(n_features),
            nn.LeakyReLU(),
        )
        self.final_layer = nn.Linear(n_features, n_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = self.final_layer(out)
        return out
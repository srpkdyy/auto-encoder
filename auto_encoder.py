import torch
import torch.nn as nn


def make_encode_layers(n_in, n_out):
    layers = [
        nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(n_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2)
    ]
    return nn.Sequential(*layers)


def make_decode_layers(n_in, n_out):
    layers = [
        nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2),
        nn.BatchNorm2d(n_out),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)

    
class AutoEncoder(nn.Module):
    def __init__(self, n_channels=3, dims=(16, 32, 64, 128)):
        super().__init__()
        self.encoder = nn.Sequential(
            make_encode_layers(n_channels, dims[0]),
            make_encode_layers(dims[0], dims[1]),
            make_encode_layers(dims[1], dims[2]),
            make_encode_layers(dims[2], dims[3])
        )
        self.decoder = nn.Sequential(
            make_decode_layers(dims[3], dims[2]),
            make_decode_layers(dims[2], dims[1]),
            make_decode_layers(dims[1], dims[0]),
            make_decode_layers(dims[0], n_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.sigmoid(out)
        return out
    
    def encode(self, x):
        out = self.encoder(x)
        out = torch.flatten(out, 1)
        return out


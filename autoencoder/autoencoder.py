import torch.nn as nn
import torch.nn.functional as F
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, encoded_dim=128):
        super(Autoencoder, self).__init__()
        decoder_layers = []
        encoder_layers = []
        encoder_dims = [512,256,encoded_dim]
        decoder_dims = [256,512,input_dim]
        for i in range(len(encoder_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim, encoder_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_dims[i-1], encoder_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)

        for i in range(len(decoder_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_dims[-1], decoder_dims[i]))
            else:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_dims[i-1]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_dims[i-1], decoder_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)        

    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    

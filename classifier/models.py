import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import kiui
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.hub.backbones import dinov2_vitl14_reg

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.silu(x, inplace=True)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # image encoder 
        self.backbone: DinoVisionTransformer = dinov2_vitl14_reg()
        del self.backbone.mask_token # remove unused params
        # self.backbone.requires_grad_(False)

        embed_dim = self.backbone.embed_dim
        layers = 4

        # classifier head
        self.mlp = MLP((1 + layers) * embed_dim, 1, embed_dim, 3)

    def forward(self, x):
        # x: [B, 3, H, W], normalized
    
        x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(dim=1),
        ], dim=1)

        out = self.mlp(linear_input)

        return out
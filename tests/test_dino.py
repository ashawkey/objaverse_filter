import sys
sys.path.append('.')

import torch
import torchvision.transforms.functional as TF
from dinov2.models.vision_transformer import DinoVisionTransformer
import kiui

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

################## dino

vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').cuda()
# print(vitb16.patch_embed.img_size, vitb16.patch_embed.patch_size, vitb16.patch_embed.num_patches)
# DINOv1 use 16 patch size, while DINOv2 use 14 patch size...
x = torch.rand(1, 3, 512, 512).cuda() # [0, 1]
x = TF.normalize(x, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
kiui.lo(x)

with torch.no_grad():
    x = vitb16.prepare_tokens(x)
    for blk in vitb16.blocks:
        x = blk(x)
    x = vitb16.norm(x)
    kiui.lo(x) # [1, 1025, 768], this aligns with the paper!

################### dinov2

vitb14_reg: DinoVisionTransformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda()
# default is 518 img_size, 14 patch_size...

x = torch.rand(1, 3, 518, 518).cuda() # [0, 1]
x = TF.normalize(x, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
kiui.lo(x)

with torch.no_grad():
    out = vitb14_reg.forward_features(x)
    for k, v in out.items():
        print(k)
        kiui.lo(v)
import torch
import torch.nn as nn
import timm
import os

class ViTFeature(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224'):
        super(ViTFeature, self).__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.vit.head = nn.Identity()

    def forward(self, x):
        features = []
        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.concat((x, cls_token), 1)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        for blk in self.vit.blocks:
            x = blk(x)
            features.append(x)

        return features
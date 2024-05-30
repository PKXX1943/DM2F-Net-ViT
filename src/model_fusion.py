import torch
import torch.nn.functional as F
from torch import nn, Tensor

import math
from typing import Tuple, Type

import torchvision.transforms as transforms
import torchvision.models as models
from resnext import ResNeXt101
from ViT_backbone import ViTFeature


class MLPBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(num_features, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, num_features)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_heads: int,
        mlp_ratio: int = 2,
        activation: Type[nn.Module] = nn.SELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        """
        super().__init__()
        self.self_attn1 = Attention(num_features, num_heads)
        self.norm1 = nn.LayerNorm(num_features)

        self.mlp = MLPBlock(num_features, num_features * mlp_ratio, activation)
        self.norm2 = nn.LayerNorm(num_features)

    def forward(self, tokens: Tensor) -> Tensor:
        # self-attention 1
        attn_out = self.self_attn1(q=tokens, k=tokens, v=tokens)
        attn_out = self.norm1(attn_out)
        # MLP block
        mlp_out = self.mlp(attn_out)
        mlp_out = self.norm2(attn_out)
        return mlp_out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_heads: int,
        mlp_ratio: int = 2,
        activation: Type[nn.Module] = nn.SELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        """
        super().__init__()

        self.cross_attn_f_to_i = Attention(
            num_features, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm1 = nn.LayerNorm(num_features)

        self.mlp = MLPBlock(num_features, num_features * mlp_ratio, activation)
        self.norm2 = nn.LayerNorm(num_features)

        self.cross_attn_i_to_f = Attention(
            num_features, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm3 = nn.LayerNorm(num_features)

    def forward(self, queries: Tensor, keys: Tensor) -> Tuple[Tensor, Tensor]:

        # Cross attention block, img to concat_feat
        q = queries 
        k = keys 
        attn_out = self.cross_attn_i_to_f(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm2(queries)

        # Cross attention block, concat_feat to img
        q = queries 
        k = keys 
        attn_out = self.cross_attn_f_to_i(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm3(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.internal_dim = num_features // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide num_features."
        
        self.q_proj = nn.Linear(num_features, self.internal_dim)
        self.k_proj = nn.Linear(num_features, self.internal_dim)
        self.v_proj = nn.Linear(num_features, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, num_features)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x HW x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x HW x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x C_per_head x HW
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        rgb_mean = (0.485, 0.456, 0.406)
        self.mean = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.229, 0.224, 0.225)
        self.std = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)


class BaseA(nn.Module):
    def __init__(self):
        super(BaseA, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63438
        self.mean[0, 1, 0, 0] = 0.59396
        self.mean[0, 2, 0, 0] = 0.58369
        self.std[0, 0, 0, 0] = 0.16195
        self.std[0, 1, 0, 0] = 0.16937
        self.std[0, 2, 0, 0] = 0.17564

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class BaseITS(nn.Module):
    def __init__(self):
        super(BaseITS, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.63542
        self.mean[0, 1, 0, 0] = 0.59579
        self.mean[0, 2, 0, 0] = 0.58550
        self.std[0, 0, 0, 0] = 0.14470
        self.std[0, 1, 0, 0] = 0.14850
        self.std[0, 2, 0, 0] = 0.15348

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False


class Base_OHAZE(nn.Module):
    def __init__(self):
        super(Base_OHAZE, self).__init__()
        rgb_mean = (0.47421, 0.50878, 0.56789)
        self.mean_in = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.10168, 0.10488, 0.11524)
        self.std_in = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)

        rgb_mean = (0.35851, 0.35316, 0.34425)
        self.mean_out = nn.Parameter(torch.Tensor(rgb_mean).view(1, 3, 1, 1), requires_grad=False)
        rgb_std = (0.16391, 0.16174, 0.17148)
        self.std_out = nn.Parameter(torch.Tensor(rgb_std).view(1, 3, 1, 1), requires_grad=False)





class DM2FNet_fusion(Base):
    def __init__(self, num_features=128, arch='resnext101_32x8d', patch_size=16):
        super(DM2FNet_fusion, self).__init__()
        self.num_features = num_features
        self.patch_size = 16

        assert arch in ['resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        resnet = models.__dict__[arch](pretrained=True)
        del resnet.fc
        self.resnet = resnet

        vit = ViTFeature()
        
        self.vit = vit

        self.down1 = nn.Sequential(
            nn.Conv2d(256, num_features, kernel_size=1), nn.SELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, num_features, kernel_size=1), nn.SELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, num_features, kernel_size=1), nn.SELU()
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, num_features, kernel_size=1), nn.SELU()
        )

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.attention_phy = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.attention1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )
        self.attention4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features * 4, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1)
        )

        self.j1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j4 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 18, kernel_size=1)
        )


        self.down_a = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=2), nn.SELU(),
            nn.ConvTranspose2d(768, 224, kernel_size=2, stride=2),
            LayerNorm2d(224),  nn.SELU(), 
            nn.ConvTranspose2d(224, num_features, kernel_size=2, stride=2),
            LayerNorm2d(num_features), nn.SELU(),
            nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2),
            LayerNorm2d(num_features), nn.SELU(),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.down_t = nn.Sequential(
            nn.Conv2d(768*2, 768, kernel_size=3, padding=2), nn.SELU(),
            nn.ConvTranspose2d(768, 224, kernel_size=2, stride=2),
            LayerNorm2d(224),  nn.SELU(), 
            nn.ConvTranspose2d(224, num_features, kernel_size=2, stride=2),
            LayerNorm2d(num_features), nn.SELU(),
            nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2),
            LayerNorm2d(num_features), nn.SELU(),
            
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )

        self.PatchEmbed = nn.Sequential(
            nn.Conv2d(3, 768, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
        )

        self.SA = nn.ModuleList()
        for i in range(4):
            self.SA.append(SelfAttentionBlock(768, num_heads=2))

        self.CA = nn.ModuleList()
        for i in range(2):
            self.CA.append(CrossAttentionBlock(768, num_heads=4))

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0, x0_hd=None):

        x = (x0 - self.mean) / self.std

        resnet = self.resnet
        vit = self.vit

        ''' ViT Begin '''
        resize_t = transforms.Resize((224, 224))
        x_attn = resize_t(x0)
        x_attn = (x_attn - self.mean) / self.std

        with torch.no_grad():

            b, c, h, w = x_attn.size()

            attn_out = vit(x_attn)

        attn1 = attn_out[5][:, 1:, :]
        attn2 = attn_out[11][:, 1:, :]

        a_embedding = self.PatchEmbed(x_attn).permute(0, 2, 3, 1).view(b, 196, 768) 

        attns = [attn1, attn2]

        for i in range(2):
            self_attn1 = self.SA[i]
            self_attn2 = self.SA[i+2]
            cross_attn = self.CA[i]
            attn = attns[i]

            a_embedding = self_attn2(self_attn1(a_embedding))

            a_embedding, attn = cross_attn(queries=a_embedding, keys=attn)
            
            attns.append(attn)

        attn1 = attns[2].permute(0, 2, 1).view(b, 768, 14, 14)
        attn2 = attns[3].permute(0, 2, 1).view(b, 768, 14, 14)
        concat_attn = torch.cat((attn1, attn2), 1)

        t_vit = self.down_t(concat_attn)
        t_vit = F.upsample(t_vit, size=x0.size()[2:], mode='bilinear')
        a_vit = self.down_a(a_embedding.permute(0, 2, 1).view(b, 768, 14, 14))  

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std
        
        # J2 = I * R1
        vit_out = ((x0 - a_vit * (1 - t_vit)) / t_vit.clamp(min=1e-8)).clamp(min=0., max=1.)
        

        ''' ViT End '''

        layer0 = resnet.conv1(x)
        layer0 = resnet.bn1(layer0)
        layer0 = resnet.relu(layer0)
        layer0 = resnet.maxpool(layer0)
        layer1 = resnet.layer1(layer0)
        layer2 = resnet.layer2(layer1)
        layer3 = resnet.layer3(layer2)
        layer4 = resnet.layer4(layer3)

        # layer0 = self.layer0(x)
        # layer1 = self.layer1(layer0)
        # layer2 = self.layer2(layer1)
        # layer3 = self.layer3(layer2)
        # layer4 = self.layer4(layer3)

        down1 = self.down1(layer1)
        down2 = self.down2(layer2)
        down3 = self.down3(layer3)
        down4 = self.down4(layer4)

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')

        concat = torch.cat((down1, down2, down3, down4), 1)

        n, c, h, w = down1.size()

        attention_phy = self.attention_phy(concat)
        attention_phy = F.softmax(attention_phy.view(n, 4, c, h, w), 1)
        f_phy = down1 * attention_phy[:, 0, :, :, :] + down2 * attention_phy[:, 1, :, :, :] + \
                down3 * attention_phy[:, 2, :, :, :] + down4 * attention_phy[:, 3, :, :, :]
        f_phy = self.refine(f_phy) + f_phy 

        attention1 = self.attention1(concat)
        attention1 = F.softmax(attention1.view(n, 4, c, h, w), 1)
        f1 = down1 * attention1[:, 0, :, :, :] + down2 * attention1[:, 1, :, :, :] + \
             down3 * attention1[:, 2, :, :, :] + down4 * attention1[:, 3, :, :, :]
        f1 = self.refine(f1) + f1 

        attention2 = self.attention2(concat)
        attention2 = F.softmax(attention2.view(n, 4, c, h, w), 1)
        f2 = down1 * attention2[:, 0, :, :, :] + down2 * attention2[:, 1, :, :, :] + \
             down3 * attention2[:, 2, :, :, :] + down4 * attention2[:, 3, :, :, :]
        f2 = self.refine(f2) + f2 

        attention3 = self.attention3(concat)
        attention3 = F.softmax(attention3.view(n, 4, c, h, w), 1)
        f3 = down1 * attention3[:, 0, :, :, :] + down2 * attention3[:, 1, :, :, :] + \
             down3 * attention3[:, 2, :, :, :] + down4 * attention3[:, 3, :, :, :]
        f3 = self.refine(f3) + f3 

        attention4 = self.attention4(concat)
        attention4 = F.softmax(attention4.view(n, 4, c, h, w), 1)
        f4 = down1 * attention4[:, 0, :, :, :] + down2 * attention4[:, 1, :, :, :] + \
             down3 * attention4[:, 2, :, :, :] + down4 * attention4[:, 3, :, :, :]
        f4 = self.refine(f4) + f4

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(f_phy) 
        t = F.upsample(self.t(f_phy), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)

        # J2 = I * exp(R2)
        r1 = F.upsample(self.j1(f1), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)

        # J2 = I + R2
        r2 = F.upsample(self.j2(f2), size=x0.size()[2:], mode='bilinear')
        x_j2 = ((x + r2) * self.std + self.mean).clamp(min=0., max=1.)

        #
        r3 = F.upsample(self.j3(f3), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)

        # J4 = log(1 + I * R4)
        r4 = F.upsample(self.j4(f4), size=x0.size()[2:], mode='bilinear')
        # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)

        attention_fusion = F.upsample(self.attention_fusion(concat), size=x0.size()[2:], mode='bilinear')
        x_f0 = torch.sum(F.softmax(attention_fusion[:, :6, :, :], 1) *
                         torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                                      x_j3[:, 0, :, :], x_j4[:, 0, :, :], vit_out[:, 0, :, :]), 1), 1, True) 
        x_f1 = torch.sum(F.softmax(attention_fusion[:, 6: 12, :, :], 1) *
                         torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                                      x_j3[:, 1, :, :], x_j4[:, 1, :, :], vit_out[:, 0, :, :]), 1), 1, True)
        x_f2 = torch.sum(F.softmax(attention_fusion[:, 12:, :, :], 1) *
                         torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                                      x_j3[:, 2, :, :], x_j4[:, 2, :, :], vit_out[:, 0, :, :]), 1), 1, True)
        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)

        if self.training:
            return x_fusion, x_j1, x_j2, x_j3, x_j4, vit_out
        else:
            return x_fusion



import torch
import torch.nn.functional as F
from torch import Tensor, nn

import math
from typing import Tuple, Type

import torchvision.transforms as transforms
import torchvision.models as models
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

class DM2FNet_CA(Base):
    def __init__(self, num_features=64, patch_size=16):
        super(DM2FNet_CA, self).__init__()
        self.num_features = num_features
        self.patch_size = patch_size

        backbone = ViTFeature()
        
        self.backbone = backbone
        
        self.down1 = nn.Sequential(
            nn.Conv2d(768, num_features, kernel_size=1),
            LayerNorm2d(num_features), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            LayerNorm2d(num_features),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(768, num_features, kernel_size=1),
            LayerNorm2d(num_features), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            LayerNorm2d(num_features),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(768, num_features, kernel_size=1),
            LayerNorm2d(num_features), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            LayerNorm2d(num_features),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(768, num_features, kernel_size=1),
            LayerNorm2d(num_features), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            LayerNorm2d(num_features),
        )
        self.down = nn.Sequential(
            nn.Conv2d(768, num_features, kernel_size=1),
            LayerNorm2d(num_features), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            LayerNorm2d(num_features),
        )

        self.PatchEmbed = nn.Sequential(
            nn.Conv2d(3, 768, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
        )

        self.SA = nn.ModuleList()
        for i in range(4):
            self.SA.append(SelfAttentionBlock(768, num_heads=2))

        self.CA = nn.ModuleList()
        for i in range(4):
            self.CA.append(CrossAttentionBlock(768, num_heads=4))

        self.A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.T = nn.Sequential(
            nn.ConvTranspose2d(num_features*4, num_features*2, kernel_size=2, stride=2),
            LayerNorm2d(num_features*2),  nn.SELU(), 
            nn.ConvTranspose2d(num_features*2, num_features, kernel_size=2, stride=2),
            LayerNorm2d(num_features), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )
        
        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x0, x0_hd=None):
        
        resize_t = transforms.Resize((224, 224))
        x = resize_t(x0)
        x = (x - self.mean) / self.std

        with torch.no_grad():
            backbone = self.backbone

        b, c, h, w = x.size()

        attn_out = backbone(x)

        layer1 = attn_out[2][:, 1:, :]
        layer2 = attn_out[5][:, 1:, :]
        layer3 = attn_out[8][:, 1:, :]
        layer4 = attn_out[11][:, 1:, :]

        a_embedding = self.PatchEmbed(x).permute(0, 2, 3, 1).view(b, 196, 768) 

        layers = [layer1, layer2, layer3, layer4]
        downs = []

        for i in range(4):
            self_attn = self.SA[i]
            cross_attn = self.CA[i]
            layer = layers[i]

            a_embedding = self_attn(a_embedding)

            a_embedding, layer = cross_attn(queries=a_embedding, keys=layer)
            
            downs.append(layer)

        down1 = self.down1(downs[0].permute(0, 2, 1).view(b, 768, 14, 14))
        down2 = self.down2(downs[1].permute(0, 2, 1).view(b, 768, 14, 14))
        down3 = self.down3(downs[2].permute(0, 2, 1).view(b, 768, 14, 14))
        down4 = self.down4(downs[3].permute(0, 2, 1).view(b, 768, 14, 14))

        t_embedding = torch.cat((down1, down2, down3, down4), 1)
        a_embedding = self.down(a_embedding.permute(0, 2, 1).view(b, 768, 14, 14))

        a = self.A(a_embedding)
        t = self.T(t_embedding)
        t = F.upsample(t, size=x0.size()[2:], mode='bilinear')

        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        # J0 = (I - A0 * (1 - T0)) / T0
        out = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)
        
        if self.training:
            return out
        else:
            return out



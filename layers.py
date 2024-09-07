import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange
from torch.nn.common_types import _size_2_t

class Linear(nn.Linear):
    """ Linear layers with different initialization methods, default is kaiming uniform (PyTorch default) """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device = None,
                 dtype = None, init_method: str = 'kaiming_uniform_') -> None:
        self.init_method = init_method
        assert self.init_method in ['kaiming_uniform_', 'xavier_uniform_', 'trunc_normal_', 'srt_'], "Invalid initialization method!"
        super(Linear, self).__init__(in_features, out_features, bias, device, dtype)

    def reset_parameters(self) -> None:
        if self.init_method == 'kaiming_uniform_':
            init.kaiming_uniform_(self.weight, a = math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)
        elif self.init_method == 'xavier_uniform_':
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias, std = 1e-6)
        elif self.init_method == 'trunc_normal_':
            input_size = self.weight.shape[-1]
            std = math.sqrt(1 / input_size)
            init.trunc_normal_(self.weight, std = std, a = -2. * std, b = 2. * std)
            if self.bias is not None:
                init.zeros_(self.bias)
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0., attn_drop: float = 0.,
                 selfatt: bool = True, kv_dim = None, init_method: str = 'kaiming_uniform_') -> None:
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim = -1)
        if selfatt:
            self.to_qkv = Linear(dim, inner_dim * 3, bias = False, init_method = init_method)
        else:
            self.to_q = Linear(dim, inner_dim, bias = False)
            self.to_kv = Linear(kv_dim, inner_dim * 2, bias = False, init_method = init_method)

        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            Linear(inner_dim, dim, init_method = init_method),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: Tensor, z = None) -> Tensor:
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim = -1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0., init_method: str = 'kaiming_uniform_') -> None:
        super().__init__()
        self.net = nn.Sequential(
            Linear(dim, hidden_dim, init_method = init_method),
            nn.GELU(),
            #nn.Dropout(dropout),
            Linear(hidden_dim, dim, init_method = init_method),
            nn.Dropout(dropout)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)

class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim = None, out_dim = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        if out_dim is None:
            out_dim = 2 * hidden_dim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, stride = 1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, stride = 2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, dim_head: int, mlp_ratio: float, dropout: float = 0., attn_drop: float = 0.,
                 selfatt: bool = True, kv_dim = None, init_method: str = 'kaiming_uniform_') -> Tensor:
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads = num_heads, dim_head = dim_head, dropout = dropout, attn_drop = attn_drop,
                                       selfatt = selfatt, kv_dim = kv_dim, init_method = init_method)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, init_method = init_method))
            ]))

    def forward(self, x, z = None):
        for attn, ff in self.layers:
            x = attn(x, z = z) + x
            x = ff(x) + x
        return x

class PositionalEncoding(nn.Module):
    """
    Generates sinusoidal positional encoding.

    Args:
        sequence_length: Length of the input sequence.
        positional_encoding_dim: Dimension of the positional encoding.
    """
    def __init__(self, sequence_length: int, positional_encoding_dim: int, dropout: float = 0.):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(sequence_length, positional_encoding_dim)
        position = torch.arange(0, sequence_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, positional_encoding_dim, 2).float() * (-math.log(10000.0) / positional_encoding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, positional_encoding_dim)

        Returns:
            Enriched input tensor with positional encoding applied
        """

        x = x + self.pe[:, :x.shape[1]].to(x.device).requires_grad_(False) 
        return self.dropout(x)

class PatchEmbeddingWithRays(nn.Module):
    """
    Embedding rays and then concatenate to the input of encoder

    Args:
        in_channels: Number of channels of ray_origins + ray_directions + imgs
        embed_dim: Dimension of embedding.
        kernel_size: Kernel size of CNN to projection the input B, C, H, W to B, C, H // patch_h, W // patch_w
    """
    def __init__(self, in_channels: int = 9, embed_dim: int = 768, kernel_size: _size_2_t = [16, 10], stride: _size_2_t = [16, 10]):
        super().__init__()

        self.proj = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, kernel_size = kernel_size, stride = stride)

    def forward(self, imgs: Tensor, ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
        B, C, H, W = imgs.shape
        
        assert imgs.shape == ray_origins.shape == ray_directions.shape

        x = torch.cat((imgs, ray_origins, ray_directions), 1)
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x

class PatchEmbeddingWithRaysDecoder(nn.Module):
    """
    Embedding rays and then add to the input of decoder

    Args:
        in_channels: Number of channels of rays
        embed_dim: Dimension of embedding.
        kernel_size: Kernel size of CNN to projection the input B, C, H, W to B, C, 1, 1
    """
    def __init__(self, in_channels: int = 6, embed_dim: int = 512, kernel_size: _size_2_t = [16, 10], stride: _size_2_t = [16, 10]):
        super().__init__()
        # self.downsample = nn.avgpool2d(kernel_size = 2, stride = 2)
        self.proj = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, kernel_size = kernel_size, stride = stride)

    def forward(self, x: Tensor, ray_origins: Tensor, ray_directions: Tensor, ids_mask: torch.long) -> Tensor:
        """
        Args:
            x: Input tensor of shape (num_tokens, embed_dim).

        Returns:
            Adding ray embeddings to input.
        """

        rays = torch.cat((ray_origins, ray_directions), 1).requires_grad_(False)
        # rays = self.downsample(rays)
        with torch.no_grad():
            rays_embedding = self.proj(rays).flatten(2).transpose(1, 2)     
            rays_embedding = torch.gather(rays_embedding, dim = 1, index = ids_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])).requires_grad_(False)
            x = x + rays_embedding.to(x.device)
        
        return x
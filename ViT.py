from layers import TransformerBlock, PositionalEncoding, PatchEmbeddingWithRays
from torch.nn.common_types import _size_2_t
import torch
import torch.nn as nn
from torch import Tensor

class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, img_size: _size_2_t = [160, 90], patch_size: _size_2_t = [40, 30], in_channels: int = 9, mask_ratio: float = 0.75,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4., drop_rate: float = 0.,
                 attn_drop_rate: float = 0., init_method = 'xavier_uniform_', use_learnable_pos_emb: bool = False, latent_dim: int = 512) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.depth = depth
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbeddingWithRays(in_channels = in_channels, embed_dim = embed_dim)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        else:
            self.pos_embed = PositionalEncoding(self.num_patches, embed_dim)

        self.blocks = TransformerBlock(dim = embed_dim, num_heads = num_heads, dim_head = embed_dim // num_heads, mlp_ratio = mlp_ratio, 
                                       dropout = drop_rate, depth = depth, attn_drop = attn_drop_rate, init_method = init_method)

        self.norm =  nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim) if embed_dim != latent_dim else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std = .02)

        self.apply(self._init_weights)
    
    def _init_weights(self, m) ->None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return depth

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device = x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim = 1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim = 1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim = 1, index = ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device = x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim = 1, index = ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, patches: Tensor, ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
        B, C, H, W = patches.shape
        x = self.patch_embed(patches, ray_origins, ray_directions)
        
        x = x + self.pos_embed(x).type_as(x).to(x.device).clone().detach()

        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
  
        x_masked = self.blocks(x_masked)
        x_masked = self.norm(x_masked)
        x_masked = self.head(x_masked)
        return x_masked, mask, ids_restore

class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, out_dim: int = 40 * 30 * 3, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4., drop_rate: float = 0.,
                 attn_drop_rate: float = 0., init_method = 'xavier_uniform_') -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth

        self.blocks = TransformerBlock(dim = embed_dim, num_heads = num_heads, dim_head = embed_dim // num_heads, mlp_ratio = mlp_ratio,
                                       dropout = drop_rate, depth = depth, attn_drop = attn_drop_rate, init_method = init_method)

        self.norm =  nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return depth

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size = [160, 90], 
                 patch_size = [40, 30], 
                 encoder_in_chans = 9,  
                 encoder_embed_dim = 768, 
                 encoder_depth = 12,
                 encoder_num_heads = 12, 
                 decoder_embed_dim = 512, 
                 decoder_depth = 8,
                 decoder_num_heads = 8, 
                 mlp_ratio = 4., 
                 qkv_bias = False, 
                 qk_scale = None, 
                 drop_rate = 0., 
                 attn_drop_rate = 0.,
                 use_learnable_pos_emb = False,
                 mask_ratio = 0.75,
                 latent_dim = 512
                 ):
        super().__init__()

        self.encoder = PretrainVisionTransformerEncoder(
            img_size = img_size, 
            patch_size = patch_size,
            in_channels = encoder_in_chans,
            mask_ratio = mask_ratio,
            embed_dim = encoder_embed_dim, 
            depth = encoder_depth,
            num_heads = encoder_num_heads,
            mlp_ratio = mlp_ratio,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            use_learnable_pos_emb = use_learnable_pos_emb, 
            latent_dim = decoder_embed_dim)

        self.decoder = PretrainVisionTransformerDecoder(
            embed_dim = decoder_embed_dim,
            out_dim = patch_size[0] * patch_size[1] * 3,
            depth = decoder_depth, 
            num_heads = decoder_num_heads,
            mlp_ratio = mlp_ratio,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = PositionalEncoding(self.encoder.num_patches, decoder_embed_dim)

        nn.init.trunc_normal_(self.mask_token, std = .02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, patches: Tensor, ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
        x, mask, ids_restore = self.encoder(patches, ray_origins, ray_directions)

        B, N, C = x.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        x = self.decoder(x_full, pos_emd_mask.shape[1])

        return x
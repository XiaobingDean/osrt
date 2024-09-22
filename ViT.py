from layers import TransformerBlock, PositionalEncoding, PatchEmbeddingWithRays
from torch.nn.common_types import _size_2_t
import torch
import torch.nn as nn
from torch import Tensor

class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int = 9, mask_ratio: float = 0.75, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, 
                 mlp_ratio: float = 4., drop_rate: float = 0., attn_drop_rate: float = 0., init_method = 'xavier_uniform_', use_learnable_pos_emb: bool = False) -> None:
        super().__init__()

        # self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.depth = depth
        self.mask_ratio = mask_ratio

        # self.patch_embed = PatchEmbeddingWithRays(in_channels = in_channels, embed_dim = embed_dim, kernel_size = patch_size)

        # if use_learnable_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        # else:
        #     self.pos_embed = PositionalEncoding(self.seq_length, embed_dim)

        self.blocks = TransformerBlock(dim = embed_dim, num_heads = num_heads, dim_head = embed_dim // num_heads, mlp_ratio = mlp_ratio, 
                                       dropout = drop_rate, depth = depth, attn_drop = attn_drop_rate, init_method = init_method)

        self.norm = nn.LayerNorm(embed_dim)

        # if use_learnable_pos_emb:
        #     trunc_normal_(self.pos_embed, std = .02)

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

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, out_dim: int = 16 * 10 * 3, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4., drop_rate: float = 0.,
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

# class PretrainVisionTransformer(nn.Module):
#     def __init__(self,
#                  img_size = [160, 90], 
#                  patch_size = [16, 10], 
#                  encoder_in_chans = 9,  
#                  encoder_embed_dim = 768, 
#                  encoder_depth = 12,
#                  encoder_num_heads = 12, 
#                  decoder_embed_dim = 512, 
#                  decoder_depth = 8,
#                  decoder_num_heads = 8, 
#                  mlp_ratio = 4., 
#                  drop_rate = 0., 
#                  attn_drop_rate = 0.,
#                  use_learnable_pos_emb = False,
#                  mask_ratio = 0.75,
#                  latent_dim = 512
#                  ):
#         super().__init__()

#         self.encoder = PretrainVisionTransformerEncoder(
#             img_size = img_size, 
#             patch_size = patch_size,
#             in_channels = encoder_in_chans,
#             mask_ratio = mask_ratio,
#             embed_dim = encoder_embed_dim, 
#             depth = encoder_depth,
#             num_heads = encoder_num_heads,
#             mlp_ratio = mlp_ratio,
#             drop_rate = drop_rate,
#             attn_drop_rate = attn_drop_rate,
#             use_learnable_pos_emb = use_learnable_pos_emb, 
#             latent_dim = decoder_embed_dim)

#         self.decoder = PretrainVisionTransformerDecoder(
#             embed_dim = decoder_embed_dim,
#             out_dim = patch_size[0] * patch_size[1] * 3,
#             depth = decoder_depth, 
#             num_heads = decoder_num_heads,
#             mlp_ratio = mlp_ratio,
#             drop_rate = drop_rate,
#             attn_drop_rate = attn_drop_rate)

#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

#         self.pos_embed = PositionalEncoding(self.encoder.num_patches, decoder_embed_dim)

#         nn.init.trunc_normal_(self.mask_token, std = .02)


#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, patches: Tensor, ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
#         x, mask, ids_restore = self.encoder(patches, ray_origins, ray_directions)
#         mask = mask.to(torch.long)

#         B, N, C = x.shape
#         # we don't unshuffle the correct visible token order, 
#         # but shuffle the pos embedding accorddingly.
#         expand_pos_embed = self.pos_embed(x).expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
#         pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
#         pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
#         x_full = torch.cat([x + pos_emd_vis, self.mask_token + pos_emd_mask], dim = 1)
#         x = self.decoder(x_full, pos_emd_mask.shape[1])

#         return x
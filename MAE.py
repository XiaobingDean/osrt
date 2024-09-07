from ViT import PretrainVisionTransformerEncoder, PretrainVisionTransformerDecoder
from layers import PatchEmbeddingWithRays, PatchEmbeddingWithRaysDecoder
from torch.nn.common_types import _size_2_t
import torch
import torch.nn as nn
from torch import Tensor

class SceneMAE(nn.Module):
    def __init__(self,
                 img_size = [160, 90], 
                 patch_size = [16, 10], 
                 encoder_in_channels = 9,  
                 encoder_embed_dim = 768, 
                 encoder_depth = 12,
                 encoder_num_heads = 12, 
                 decoder_embed_dim = 512, 
                 decoder_depth = 8,
                 decoder_num_heads = 8, 
                 mlp_ratio = 4., 
                 drop_rate = 0., 
                 attn_drop_rate = 0.,
                 use_learnable_pos_emb = False,
                 mask_ratio = 0.75,
                 ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_h = img_size[0] // patch_size[0]
        self.patch_w = img_size[1] // patch_size[1]
        self.num_patches = self.patch_h * self.patch_w
        self.mask_ratio = mask_ratio

        self.encoder = PretrainVisionTransformerEncoder(
            in_channels = encoder_in_channels,
            embed_dim = encoder_embed_dim, 
            depth = encoder_depth,
            num_heads = encoder_num_heads,
            mlp_ratio = mlp_ratio,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate,
            use_learnable_pos_emb = use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            embed_dim = decoder_embed_dim,
            out_dim = patch_size[0] * patch_size[1] * 3,
            depth = decoder_depth, 
            num_heads = decoder_num_heads,
            mlp_ratio = mlp_ratio,
            drop_rate = drop_rate,
            attn_drop_rate = attn_drop_rate)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.patch_embed = PatchEmbeddingWithRays(in_channels = 9, embed_dim = encoder_embed_dim, kernel_size = patch_size, stride = patch_size)
        self.patch_embed_decoder = PatchEmbeddingWithRaysDecoder(in_channels = 6, embed_dim = decoder_embed_dim, kernel_size = patch_size, stride = patch_size)
        self.linear = nn.Linear(encoder_embed_dim, decoder_embed_dim) if encoder_embed_dim != decoder_embed_dim else nn.Identity()
        # self.pos_embed = PositionalEncoding(self.encoder.num_patches, decoder_embed_dim)

        nn.init.trunc_normal_(self.mask_token, std = .02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        imgs: (N, 3, H, W)
        output: (N, L, patch_h * patch_w *3)
        """

        imgs = imgs.reshape(shape = (imgs.shape[0], 3, self.patch_h, self.patch_size[0], self.patch_w, self.patch_size[1]))
        imgs = torch.einsum('nchpwq->nhwpqc', imgs)
        imgs = imgs.reshape(shape= (imgs.shape[0], self.patch_h * self.patch_w, self.patch_size[0] * self.patch_size[1] * 3))
        return imgs

    def unpatchify(self, patches: Tensor) -> Tensor:
        """
        x: (N, L, patch_h * patch_w *3)
        imgs: (N, 3, H, W)
        """
        
        patches = patches.reshape(shape = (patches.shape[0], self.patch_h, self.patch_w, self.patch_size[0], self.patch_size[1], 3))
        patches = torch.einsum('nhwpqc->nchpwq', patches)
        patches = patches.reshape(shape = (patches.shape[0], 3, self.img_size[0], self.img_size[1]))
        return patches

    def random_masking(self, x: Tensor, mask_ratio = None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        if mask_ratio == None:
            mask_ratio = self.mask_ratio

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device = x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim = 1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim = 1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim = 1, index = ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device = x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim = 1, index = ids_restore)

        return x_masked, mask, ids_restore, ids_mask

    def forward(self, imgs: Tensor, ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
        B, C, H, W = imgs.shape
        x = self.patch_embed(imgs, ray_origins, ray_directions)

        x_masked, mask, ids_restore, ids_mask = self.random_masking(x, self.mask_ratio)
        mask = mask.to(torch.long)
        
        x_masked = self.encoder(x_masked)
        x_masked = self.linear(x_masked)

        mask_tokens = self.mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
        mask_tokens = self.patch_embed_decoder(mask_tokens, ray_origins, ray_directions, ids_mask)
        x = torch.cat([x_masked, mask_tokens], dim = 1)

        x = torch.gather(x, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = self.decoder(x)

        return x, mask
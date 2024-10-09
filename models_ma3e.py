# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.ot import SinkhornDistance


class MaskedAngleAwareAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False, crop_size=96):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.window_size = self.crop_size // self.patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.angle_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.ot_loss = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.angle_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, patch_ids, window_location, b_ratio, r_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, C = x.shape  # batch, length, dim
        window_L = self.window_size * self.window_size

        len_window_keep = int(window_L * (1 - r_ratio))
        noise_window = torch.rand(N, window_L, device=x.device)
        window_ids_shuffle = torch.argsort(noise_window, dim=1)
        window_ids_restore = torch.argsort(window_ids_shuffle, dim=1)

        ids_window_keep = window_ids_shuffle[:, :len_window_keep]
        window_x = x[window_location].view(N, window_L, C)
        angle_embeds = self.angle_embed.repeat(N, window_L, 1)
        window_x = window_x + angle_embeds
        window_x_masked = torch.gather(window_x, dim=1, index=ids_window_keep.unsqueeze(-1).repeat(1, 1, C))

        window_mask = torch.ones([N, window_L], device=x.device)
        window_mask[:, :len_window_keep] = 0
        window_mask = torch.gather(window_mask, dim=1, index=window_ids_restore)

        noise_init = torch.zeros([N, L], device=x.device)
        # in window positions: 1 is vis  2 is mask
        noise_init = noise_init.scatter_(1, patch_ids.view(-1, window_L), window_mask + 1)
        new_noise = torch.rand(N, L, device=x.device)
        noise_all = noise_init.float() + new_noise

        all_ids_shuffle = torch.argsort(noise_all, dim=1)
        all_ids_restore = torch.argsort(all_ids_shuffle, dim=1)

        len_bg_keep = int((L - window_L) * (1 - b_ratio))
        ids_bg_keep = all_ids_shuffle[:, :len_bg_keep]
        bg_x_masked = torch.gather(x, dim=1, index=ids_bg_keep.unsqueeze(-1).repeat(1, 1, C))  # background

        mask_all = torch.ones([N, L], device=window_mask.device)
        mask_all[:, :len_bg_keep] = 0
        mask_all = torch.gather(mask_all, dim=1, index=all_ids_restore)
        mask_all = mask_all.scatter_(1, patch_ids.view(-1, window_L), window_mask)
        bg_mask = mask_all[~window_location].view(N, -1)

        x_masked = torch.cat((window_x_masked, bg_x_masked), dim=1)

        return x_masked, mask_all, (window_mask, bg_mask), all_ids_restore

    def generate_rotated_crop_patches_ids(self, x, tops, lefts, window_number):  # , mask_ratio):
        N, H, W, _ = x.shape

        #  extract the windows based on the coordinates
        lefts = lefts.unsqueeze(-1).expand(-1, window_number, self.window_size)  # N, 1 -> N, 1, 6
        tops = tops.unsqueeze(-1).expand(-1, window_number, self.window_size)

        row_range = torch.arange(0, self.window_size, device=x.device).unsqueeze(0).unsqueeze(0)
        rows = row_range.expand(N, window_number, self.window_size) + lefts  # N, 1, 6
        column_range = torch.arange(0, self.window_size * W, W, device=x.device).unsqueeze(0).unsqueeze(0)
        columns = column_range.expand(N, window_number, self.window_size) + tops * W  # N, 1, 6

        patches_in_window = rows.unsqueeze(2).expand(N, window_number, self.window_size, self.window_size) + \
                            columns.unsqueeze(-1).expand(N, window_number, self.window_size, self.window_size)  # N, 1, 6, 6
        window_patches = patches_in_window.view(N, window_number, -1)  # N, 1, 36

        sorted_patch_to_keep, _ = torch.sort(window_patches, dim=-1)

        return sorted_patch_to_keep

    def forward_encoder(self, x, top_starts, left_starts, b_ratio, r_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, _, C = x.shape
        H = W = self.img_size // self.patch_size
        assert top_starts.shape[1] == left_starts.shape[1]
        window_number = top_starts.shape[1]
        x = x.view(N, H, W, C)  # B, 14, 14, 768
        assert self.window_size <= H and self.window_size <= W

        # generate the sampled and mask patches from the small windows
        patch_ids = self.generate_rotated_crop_patches_ids(x, top_starts, left_starts, window_number)

        window_location = torch.zeros(N, window_number, H * W, device=patch_ids.device).to(torch.int64)
        window_location = window_location.scatter_(2, patch_ids, 1).view(-1, H * W).to(torch.bool)
        x = x.view(N, H * W, C).unsqueeze(1).repeat(1, window_number, 1, 1).view(N * window_number, H * W, C)

        # masking: length -> length * mask_ratio
        x, all_mask, masks, ids_restore = self.random_masking(x, patch_ids, window_location, b_ratio, r_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, window_location, masks, ids_restore

    def forward_decoder(self, x, window_location, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        N, _, C = x.shape
        H = W = self.img_size // self.patch_size
        patch_num = H * W

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, patch_num + 1 - x.shape[1], 1)  # N, 147, C
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        window_x_ = x_[window_location].view(N, -1, C)
        bg_x_ = x_[~window_location].view(N, -1, C)

        expand_pos_embed = self.decoder_pos_embed[:, 1:, :].expand(N, -1, -1)
        pos_window = expand_pos_embed[window_location].view(N, -1, C)
        pos_bg = expand_pos_embed[~window_location].view(N, -1, C)

        x_ = torch.cat([window_x_ + pos_window, bg_x_ + pos_bg], dim=1)
        # append cls token
        x = torch.cat([x[:, :1, :] + self.decoder_pos_embed[:, :1, :], x_], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, pos_bg.shape[1]

    def forward_loss(self, imgs, window_location, bg_num, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        N, _, C = pred.shape
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        bg_target = target[~window_location].view(N, -1, C)
        loss_bg = (pred[:, -bg_num:, :] - bg_target) ** 2
        loss_bg = loss_bg.mean(dim=-1)  # [N, L], mean loss per patch
        loss_bg = (loss_bg * mask[1]).sum() / mask[1].sum()  # mean loss on removed patches

        window_target = target[window_location].view(N, -1, C)
        window_target_norm = window_target / window_target.norm(dim=2, keepdim=True)
        window_pred = pred[:, :-bg_num, :] / pred[:, :-bg_num, :].norm(dim=2, keepdim=True)
        loss_window = self.ot_loss(window_pred, window_target_norm)

        return loss_bg + loss_window

    def forward(self, imgs, ori_imgs, tops, lefts, b_ratio=0.75, r_ratio=0.75):
        latent, locations, masks, ids_restore = self.forward_encoder(imgs, tops, lefts, b_ratio, r_ratio)
        pred, bg_num = self.forward_decoder(latent, locations, ids_restore)  # [N, L, p*p*3], 160
        loss = self.forward_loss(ori_imgs, locations, bg_num, pred, masks)
        return loss, pred, masks


def ma3e_vit_small_patch16_dec384d8b(**kwargs):
    model = MaskedAngleAwareAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ma3e_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAngleAwareAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ma3e_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAngleAwareAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ma3e_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAngleAwareAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
ma3e_vit_small_patch16 = ma3e_vit_small_patch16_dec384d8b  # decoder: 384 dim, 8 blocks
ma3e_vit_base_patch16 = ma3e_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ma3e_vit_large_patch16 = ma3e_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
ma3e_vit_huge_patch14 = ma3e_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


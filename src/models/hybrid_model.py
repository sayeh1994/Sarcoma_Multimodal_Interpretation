import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


import torch
import torch.nn.functional as F


##############################################################
# Collate function for TitanHybridMIL exporter
##############################################################

def hybrid_collate_fn(batch):

    slide_features = torch.stack(
        [item["slide_features"] for item in batch]
    )

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long
    )

    slide_ids = [item["slide_id"] for item in batch]
    paths = [item["path"] for item in batch]

    max_n = max(
        item["patch_features"].shape[0]
        for item in batch
    )

    patch_list = []
    coord_list = []
    mask_list = []
    patch_index_list = []
    n_valid_list = []

    for item in batch:

        patch = item["patch_features"]
        coord = item["coords"]
        patch_idx = item["patch_indices"]

        n = patch.shape[0]
        pad_len = max_n - n

        patch = F.pad(
            patch,
            (0, 0, 0, pad_len)
        )

        coord = F.pad(
            coord,
            (0, 0, 0, pad_len)
        )

        padded_patch_idx = F.pad(
            patch_idx,
            (0, pad_len),
            value=-1
        )

        mask = torch.zeros(
            max_n,
            dtype=torch.bool
        )
        mask[n:] = True

        patch_list.append(patch)
        coord_list.append(coord)
        patch_index_list.append(padded_patch_idx)
        mask_list.append(mask)
        n_valid_list.append(n)

    patch_features = torch.stack(patch_list)
    coords = torch.stack(coord_list)
    patch_indices = torch.stack(patch_index_list)
    mask = torch.stack(mask_list)

    batch_dict = {

        "patch_features": patch_features,
        "coords": coords,
        "slide_features": slide_features,
        "mask": mask,
        "labels": labels,
        "slide_ids": slide_ids,
        "paths": paths,
        "patch_indices": patch_indices,
        "n_valid": n_valid_list
    }

    return batch_dict


def cls_attention_to_patch_scores(
    attn,
    n_valid
):
    """
    attn: tensor of shape (B, heads, T, T)
    n_valid: number of real patches before padding
    """

    ##########################################################
    # CLS token is token 0
    # patch tokens are 1..n_valid
    ##########################################################

    cls_to_patch = attn[0, :, 0, 1:n_valid+1]   # (heads, N)

    ##########################################################
    # Average over heads
    ##########################################################

    patch_scores = cls_to_patch.mean(dim=0)     # (N,)

    patch_scores = patch_scores.detach().cpu().numpy()

    return patch_scores

# COORD ENCODER
class CoordEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, coords):
        return self.mlp(coords)


# TRANSFORMER WITH ATTENTION
class TransformerLayerWithAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, key_padding_mask=None):
        attn_out, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))

        return x, attn_weights  # (B, heads, T, T)


# MODEL
class TitanHybridMIL(nn.Module):
    def __init__(self, input_dim=768, n_classes=5, n_heads=8, n_layers=2):
        super().__init__()

        self.coord_encoder = CoordEncoder(input_dim)
        self.norm = nn.LayerNorm(input_dim)

        self.layers = nn.ModuleList([
            TransformerLayerWithAttention(input_dim, n_heads)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, n_classes)
        )

    def forward(self, patch_features, coords, slide_features, mask=None, return_attention=False):

        coord_embed = self.coord_encoder(coords)
        patch_features = patch_features + coord_embed

        cls_token = slide_features.unsqueeze(1)
        tokens = torch.cat([cls_token, patch_features], dim=1)

        tokens = self.norm(tokens)

        if mask is not None:
            cls_mask = torch.zeros((mask.size(0), 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        attentions = []

        for layer in self.layers:
            tokens, attn = layer(tokens, key_padding_mask=mask)
            attentions.append(attn)

        cls_out = tokens[:, 0]
        logits = self.classifier(cls_out)

        if return_attention:
            return logits, attentions

        return logits
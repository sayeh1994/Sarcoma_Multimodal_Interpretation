import torch
import numpy as np

from src.dataset.dataloader import PatchEvidence


##############################################################
# HybridMIL inference
##############################################################

@torch.no_grad()
def infer_slide(
    model,
    sample,
    device
):
    """
    Returns:
        probs: np.ndarray of shape (n_classes,)
        attention_scores: np.ndarray of shape (N_patches,)
    """

    patch_features = sample["patch_features"].unsqueeze(0).to(device)   # (1, N, D)
    coords = sample["coords"].unsqueeze(0).to(device)                   # (1, N, 2)

    slide_features = sample["slide_features"]
    if slide_features is not None:
        slide_features = slide_features.unsqueeze(0).to(device)

    n_valid = sample["patch_features"].shape[0]

    # No padding in single-slide export, so mask is all False
    mask = torch.zeros((1, n_valid), dtype=torch.bool, device=device)

    logits, attentions = model(
        patch_features=patch_features,
        coords=coords,
        slide_features=slide_features,
        mask=mask,
        return_attention=True
    )

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ##########################################################
    # attentions is a list of layers
    # each layer attention has shape: (B, heads, T, T)
    ##########################################################

    last_attn = attentions[-1]

    ##########################################################
    # CLS token is token 0
    # patch tokens are 1..N
    ##########################################################

    cls_to_patch = last_attn[0, :, 0, 1:n_valid+1]   # (heads, N)

    ##########################################################
    # Average across heads -> one score per patch
    ##########################################################

    attention_scores = cls_to_patch.mean(dim=0).detach().cpu().numpy()

    ##########################################################
    # Normalize for readability / comparability
    ##########################################################

    attention_scores = np.exp(attention_scores - attention_scores.max())
    attention_scores = attention_scores / attention_scores.sum()

    return probs, attention_scores


##############################################################
# Rank patches while keeping original H5 patch indices
##############################################################

def rank_attention(
    attention,
    coords,
    patch_indices,
    top_k=20
):

    attention = np.asarray(attention)
    coords = np.asarray(coords)
    patch_indices = np.asarray(patch_indices)

    order = np.argsort(attention)[::-1]

    if len(attention) > 1:
        attention_percentile = (
            attention.argsort().argsort() / (len(attention) - 1)
        ) * 100
    else:
        attention_percentile = np.array([100.0])

    ##########################################################
    # Top
    ##########################################################

    top = []
    for rank, idx in enumerate(order[:top_k]):

        top.append(
            PatchEvidence(
                patch_id=f"patch_{rank+1:04d}",
                rank=rank+1,
                patch_index=int(patch_indices[idx]),
                coord_x=float(coords[idx][0]),
                coord_y=float(coords[idx][1]),
                attention=float(attention[idx]),
                percentile=float(attention_percentile[idx])
            )
        )

    ##########################################################
    # Middle
    ##########################################################

    middle = []
    mid_start = max(0, len(order)//2 - top_k//2)
    mid_end = min(len(order), mid_start + top_k)

    for rank, idx in enumerate(order[mid_start:mid_end]):

        middle.append(
            PatchEvidence(
                patch_id=f"middle_{rank+1:04d}",
                rank=rank+1,
                patch_index=int(patch_indices[idx]),
                coord_x=float(coords[idx][0]),
                coord_y=float(coords[idx][1]),
                attention=float(attention[idx]),
                percentile=float(attention_percentile[idx])
            )
        )

    ##########################################################
    # Bottom
    ##########################################################

    bottom = []
    for rank, idx in enumerate(order[-top_k:]):

        bottom.append(
            PatchEvidence(
                patch_id=f"bottom_{rank+1:04d}",
                rank=rank+1,
                patch_index=int(patch_indices[idx]),
                coord_x=float(coords[idx][0]),
                coord_y=float(coords[idx][1]),
                attention=float(attention[idx]),
                percentile=float(attention_percentile[idx])
            )
        )

    return top, middle, bottom
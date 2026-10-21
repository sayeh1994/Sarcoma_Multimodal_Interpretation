import h5py
import numpy as np
import torch
import torch.nn.functional as F


##############################################################
# Dataset for TitanHybridMIL
##############################################################

class EvidenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe,
        max_patches=None,
        patch_key="patch_features",
        normalize_features=True,
        normalize_coords=False,
        seed=0
    ):

        self.df = dataframe.reset_index(drop=True)

        self.max_patches = max_patches
        self.patch_key = patch_key
        self.normalize_features = normalize_features
        self.normalize_coords = normalize_coords
        self.rng = np.random.default_rng(seed)

    ##########################################################

    def __len__(self):

        return len(self.df)

    ##########################################################

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        with h5py.File(row["path"], "r") as f:

            slide_features = torch.tensor(
                f["features"][:],
                dtype=torch.float32
            )

            patch_features = torch.tensor(
                f[self.patch_key][:],
                dtype=torch.float32
            )

            coords = torch.tensor(
                f["coords"][:],
                dtype=torch.float32
            )

        ######################################################
        # Safety
        ######################################################

        if slide_features.ndim > 1:
            slide_features = slide_features.squeeze()

        if patch_features.ndim == 1:
            patch_features = patch_features.unsqueeze(0)

        if coords.ndim == 1:
            coords = coords.unsqueeze(0)

        ######################################################
        # Subsample patches if needed
        ######################################################

        n_patches = patch_features.shape[0]
        patch_indices = np.arange(n_patches, dtype=np.int64)

        if (
            self.max_patches is not None
            and n_patches > self.max_patches
        ):

            patch_indices = self.rng.choice(
                n_patches,
                size=self.max_patches,
                replace=False
            )

            patch_indices = np.sort(patch_indices)

            patch_features = patch_features[patch_indices]
            coords = coords[patch_indices]

        ######################################################
        # Normalize features if desired
        ######################################################

        if self.normalize_features:

            slide_features = F.normalize(
                slide_features,
                dim=-1
            )

            patch_features = F.normalize(
                patch_features,
                dim=-1
            )

        ######################################################
        # Normalize coords only if you really want to
        ######################################################

        if self.normalize_coords:

            coords_np = coords.numpy().astype(np.float32)

            coords_np = coords_np / (
                coords_np.max(axis=0, keepdims=True) + 1e-6
            )

            coords = torch.tensor(
                coords_np,
                dtype=torch.float32
            )

        sample = {

            "slide_features": slide_features,

            "patch_features": patch_features,

            "coords": coords,

            "patch_indices": torch.tensor(
                patch_indices,
                dtype=torch.long
            ),

            "label": int(row["label"]),

            "slide_id": row["slide_id"],

            "path": row["path"],

            "wsi_path": row["wsi_path"]

        }

        return sample
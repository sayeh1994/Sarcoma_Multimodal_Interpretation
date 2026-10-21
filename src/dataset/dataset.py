import torch
import h5py

##############################################################
# Dataset
##############################################################

class EvidenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        with h5py.File(row["path"], "r") as f:

            features = torch.tensor(
                f["features"][:],
                dtype=torch.float32
            )

            coords = torch.tensor(
                f["coords"][:],
                dtype=torch.float32
            )

        if features.ndim == 1:

            features = features.unsqueeze(0)
            coords = coords.unsqueeze(0)

        sample = {

            "features": features,

            "coords": coords,

            "label": int(row["label"]),

            "slide_id": row["slide_id"],

            "path": row["path"]

        }

        return sample

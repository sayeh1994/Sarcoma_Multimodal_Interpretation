'''
python3 export_mil_features.py \
--checkpoint /path/to/checkpoint.pt \
--csv /path/to/input_data.csv \ --> should contain columns: slide_id, path to the feature .h5, label and wsi_path to .svs
--output /path/to/output/
--top_k 100 \
--device cuda \
--tcga_metadata /path/to/TCGA-SARC.GDC_phenotype.tsv  --> if exists otherwise remove this line
--cptac_metadata /path/to/TCIA_CPTAC_Pathology_Portal.csv  --> if exists otherwise remove this line
--manual_organ "Uterus"  --> if exists otherwise remove this line
'''


import os
import sys
sys.path.append(os.path.abspath("/path/to/cellvit++/project/"))  # Add the path to the cellvit++ project

import argparse

from pathlib import Path
import pandas as pd
import torch

from src.dataset.metadata import MetadataManager

from src.util.builder import export_dataset

# from src.models.loader_MIL_Lab import load_model
from src.models.loader import load_model

#==========================================================
# MODEL Customization
from src.models.hybrid_model import TitanHybridMIL
# from src.dataset.dataset import EvidenceDataset
from src.dataset.dataset_hybrid import EvidenceDataset
#==========================================================



##############################################################
# Command line
##############################################################

def parse_args():

    parser = argparse.ArgumentParser(
        description="MIL Evidence Exporter"
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing slide_id, path and label"
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--top_k",
        default=20,
        type=int
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--tcga_metadata",
        default=None
    )

    parser.add_argument(
        "--cptac_metadata",
        default=None
    )

    parser.add_argument(
        "--manual_organ",
        default=None,
        help="Used only for non-TCGA/CPTAC datasets"
    )

    return parser.parse_args()

##############################################################
# Main
##############################################################

def main():

    args = parse_args()

    ##########################################################

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("="*70)
    print("MIL Evidence Exporter")
    print("="*70)
    print(f"Device : {device}")
    print()

    ##########################################################

    dataframe = pd.read_csv(
        args.csv
    )

    print(
        f"{len(dataframe)} slides loaded."
    )

    dataset = EvidenceDataset(
        dataframe=dataframe,
        max_patches=None,          # or a fixed value if you must
        patch_key="patch_features",
        normalize_features=True,
        normalize_coords=False,
        seed=0
    )

    ##########################################################

    metadata = MetadataManager(
        tcga_tsv=args.tcga_metadata,
        cptac_csv=args.cptac_metadata
    )

    ##########################################################
    # MODEL
    model = TitanHybridMIL(
        input_dim=768,
        n_classes=5,
    ).to(device)

    model = load_model(
        model=model,
        checkpoint=args.checkpoint,
        device=device

    )
    print()
    print("Model loaded.")

    ##########################################################

    Path(
        args.output
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    ##########################################################

    export_dataset(
        dataframe=dataframe,
        dataset=dataset,
        model=model,
        metadata=metadata,
        output_dir=args.output,
        device=device,
        top_k=args.top_k,
        manual_organ=args.manual_organ
    )

    ##########################################################

    print()
    print("="*70)
    print("Finished")
    print("="*70)
    print()
    print(
        f"Evidence written to: {args.output}"
    )


##############################################################

if __name__ == "__main__":
    main()
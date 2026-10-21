import os
import sys
sys.path.append(os.path.abspath("/path/to/cellvit++/project"))  # Adjust the path as needed

import torch
from pathlib import Path
from tqdm import tqdm

from src.configs.config import CLASS_NAMES
from .attention_extraction import infer_slide, rank_attention
from .helper import build_prediction, create_slide_folder, save_json

import subprocess
import time
import json
import openslide
# import openslide
from PIL import Image

from collections import Counter
import numpy as np


# Convert dataclass -> dict

def patch_to_dict(patch):

    return {
        "patch_id": patch.patch_id,
        "rank": patch.rank,
        "patch_index": patch.patch_index,
        "coord": [
            patch.coord_x,
            patch.coord_y
        ],
        "attention": patch.attention,
        "percentile": patch.percentile,

        "image_path": patch.image_path,
        "cellvit_path": patch.cellvit_path,

        "cell_summary": getattr(
            patch,
            "cellvit_summary",
            {}
        ),

        "cell_statistics":getattr(
            patch,
            "cell_statistics",
            {}
        ),

        "spatial_statistics": getattr(
            patch,
            "spatial_statistics",
            {}
        )
    }

def load_json(file_path):
    """Load a JSON file and return its contents as a Python object."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compute_cell_statistics(summary):
    total = sum(summary.values())

    if total == 0:
        return {}

    tumor = summary.get("Neoplastic", 0)
    stroma = summary.get("Connective", 0)
    immune = summary.get("Inflammatory", 0)
    epithelial = summary.get("Epithelial", 0)
    dead = summary.get("Dead", 0)

    return {
        "total_cells": total,

        "tumor_fraction": tumor / total,

        "stroma_fraction": stroma / total,

        "immune_fraction": immune / total,

        "epithelial_fraction": epithelial / total,

        "dead_fraction": dead / total,

        "tumor_stroma_ratio":
            tumor / max(stroma,1),

        "immune_to_tumor_ratio":
            immune / max(tumor,1),

        "dominant_cell_type":
            max(summary,key=summary.get),

        "cell_density":
            total
    }


def aggregate_patch_group(patches):
    """
    Aggregate CellViT statistics across a list of patches
    (e.g. Top, Middle or Bottom).
    """

    raw = Counter()

    densities = []
    tumor_fractions = []
    stroma_fractions = []
    immune_fractions = []

    tumor_stroma_ratios = []
    immune_tumor_ratios = []

    # For future spatial statistics
    cluster_scores = []
    infiltration_scores = []
    entropies = []

    valid_patches = 0

    for patch in patches:

        if not hasattr(patch, "cellvit_summary"):
            continue

        if len(patch.cellvit_summary) == 0:
            continue

        valid_patches += 1

        raw.update(patch.cellvit_summary)

        stats = patch.cell_statistics

        densities.append(stats["cell_density"])

        tumor_fractions.append(stats["tumor_fraction"])
        stroma_fractions.append(stats["stroma_fraction"])
        immune_fractions.append(stats["immune_fraction"])

        tumor_stroma_ratios.append(stats["tumor_stroma_ratio"])
        immune_tumor_ratios.append(stats["immune_to_tumor_ratio"])

        # ---------- Future spatial features ----------
        if hasattr(patch, "spatial_statistics"):

            spatial = patch.spatial_statistics

            if "tumor_cluster_score" in spatial:
                cluster_scores.append(
                    spatial["tumor_cluster_score"]
                )

            if "immune_infiltration" in spatial:
                infiltration_scores.append(
                    spatial["immune_infiltration"]
                )

            if "cell_entropy" in spatial:
                entropies.append(
                    spatial["cell_entropy"]
                )

    if valid_patches == 0:
        return {}

    # Overall slide counts
    pooled_statistics = compute_cell_statistics(dict(raw))

    patch_statistics = {

        "n_valid_patches": valid_patches,

        "mean_cell_density": float(np.mean(densities)),
        "std_cell_density": float(np.std(densities)),

        "mean_tumor_fraction": float(np.mean(tumor_fractions)),
        "std_tumor_fraction": float(np.std(tumor_fractions)),

        "mean_stroma_fraction": float(np.mean(stroma_fractions)),
        "std_stroma_fraction": float(np.std(stroma_fractions)),

        "mean_immune_fraction": float(np.mean(immune_fractions)),
        "std_immune_fraction": float(np.std(immune_fractions)),

        "mean_tumor_stroma_ratio":
            float(np.mean(tumor_stroma_ratios)),

        "mean_immune_tumor_ratio":
            float(np.mean(immune_tumor_ratios))
    }

    if len(cluster_scores):
        patch_statistics["mean_cluster_score"] = float(np.mean(cluster_scores))

    if len(infiltration_scores):
        patch_statistics["mean_immune_infiltration"] = float(np.mean(infiltration_scores))

    if len(entropies):
        patch_statistics["mean_cell_entropy"] = float(np.mean(entropies))

    return {

        "pooled_statistics": pooled_statistics,

        "patch_statistics": patch_statistics

    }

# JSON builder

def build_slide_json(
    slide_id,
    dataset,
    organ,
    ground_truth,
    prediction,
    top,
    middle,
    bottom,
    top_statistics,
    middle_statistics,
    bottom_statistics
):

    return {
        "slide": {
            "slide_id": slide_id,
            "dataset": dataset,
            "patient_id": "-".join(slide_id.split("-")[:4]),
            "organ_of_origin": organ,
            "ground_truth": ground_truth
        },

        "prediction": {
            "predicted_class": prediction.predicted_class,
            "confidence": prediction.confidence,
            "entropy": prediction.entropy,
            "margin": prediction.margin,
            "probabilities": prediction.probabilities
        },

        "slide_statistics": {

            "top": top_statistics,

            "middle": middle_statistics,

            "bottom": bottom_statistics

        },

        "attention": {
            "top": [patch_to_dict(x) for x in top],
            "middle": [patch_to_dict(x) for x in middle],
            "bottom": [patch_to_dict(x) for x in bottom]
        }
    }


# Export one slide

@torch.no_grad()
def export_slide(
    sample,
    model,
    metadata,
    output_dir,
    top_k,
    device,
    manual_organ=None
):
    os.makedirs(output_dir, exist_ok=True)
    slide_id = sample["slide_id"]
    label = sample["label"]

    evidence_path = Path(output_dir) / slide_id / "evidence.json"

    if evidence_path.exists():
        print(f"Skipping already processed slide: {slide_id}")
        return None
    # Hybrid model inference

    probs, attention = infer_slide(
        model=model,
        sample=sample,
        device=device
    )

    # Patch coordinates and original patch indices

    coords = sample["coords"].cpu().numpy()
    patch_indices = sample["patch_indices"].cpu().numpy()

    # Build prediction object

    prediction = build_prediction(probs)

    # Rank attention regions

    top, middle, bottom = rank_attention(
        attention=attention,
        coords=coords,
        patch_indices=patch_indices,
        top_k=top_k
    )
    
    roi_records = []

    for group_name, patch_list in [
        ("top", top),
        ("middle", middle),
        ("bottom", bottom),
    ]:
        for p in patch_list:
            roi_records.append({
                "patch_id": p.patch_id,
                "group": group_name,
                "rank": int(p.rank),
                "patch_index": int(p.patch_index),
                "coord_x": int(p.coord_x),
                "coord_y": int(p.coord_y),
                "attention": float(p.attention),
                "percentile": float(p.percentile),
            })

    roi_coords = [(r["coord_x"], r["coord_y"]) for r in roi_records]

    # print(sample.keys())
    wsi_path = Path(sample["wsi_path"])
    wsi_stem = slide_id.split('.svs')[0]

    if wsi_stem.upper().startswith("TCGA"):
        wsi_stem = wsi_stem.split('.')[0]

    cellvit_output_dir = Path(output_dir) / slide_id / "cellvit"
    #######################################################
    # CellViT processing
    #######################################################
    subprocess.run([
        "python",
        "CellViT-plus-plus/cellvit/detect_cells.py",
        "--model", "CellViT-plus-plus/checkpoints/SAM/CellViT-SAM-H-x40-AMP.pth",
        "--outdir", str(cellvit_output_dir),
        "process_wsi",
        "--wsi_path", str(wsi_path),
        "--roi_coords", json.dumps(roi_coords),
        "--roi_patch_size", "512"
    ], check=True)

    # time.sleep(5)  # Wait for CellViT to finish processing

    cellvit_json_path = cellvit_output_dir / f"{wsi_stem}_cells.json"
    cellvit_data = load_json(cellvit_json_path)

    from collections import defaultdict

    # ------------------------------------------------------------------
    # Group detected cells by the CellViT patch they belong to
    # ------------------------------------------------------------------
    cells_by_patch = defaultdict(list)

    for cell in cellvit_data["cells"]:
        key = tuple(cell["offset_global"])   # (x, y) of the CellViT patch
        cells_by_patch[key].append(cell)

    print(f"\nDetected {len(cellvit_data['cells'])} cells")
    print(f"Detected {len(cells_by_patch)} CellViT patches")
    print("First CellViT patch keys:", list(cells_by_patch.keys())[:10])

    # ------------------------------------------------------------------
    # Attach CellViT information to each MIL patch
    # ------------------------------------------------------------------

    slide = openslide.OpenSlide(str(wsi_path))

    for patch in top + middle + bottom:

        patch_img = slide.read_region(
            (int(patch.coord_x), int(patch.coord_y)),   # level-0 coordinates
            0,                                # highest resolution
            (512, 512)
        ).convert("RGB")

        patch_dir = Path(output_dir)/ slide_id / "patches"
        patch_dir.mkdir(exist_ok=True)

        filename = f"{patch.rank}_{patch.patch_id}.png"

        save_path = patch_dir / filename

        patch_img.save(save_path)

        patch.image_path = str(save_path)
        
        key = (patch.coord_x, patch.coord_y)

        print(
            f"MIL patch {patch.patch_id}: "
            f"coord={key} -> "
            f"{len(cells_by_patch.get(key, []))} cells"
        )

        cells = cells_by_patch.get(key, [])

        if len(cells) == 0:
            patch.cellvit_path = None
            patch.image_path = None
            patch.cellvit_summary = {}
            patch.cellvit_cells = []
            continue

        patch.cellvit_path = str(cellvit_json_path)
        # patch.image_path = None
        patch.cellvit_cells = cells

        summary = {}

        for cell in cells:
            cell_type = cellvit_data["type_map"][str(cell["type"])]
            summary[cell_type] = summary.get(cell_type, 0) + 1

        patch.cellvit_summary = summary

        patch.cell_statistics = compute_cell_statistics(summary)

        patch.spatial_statistics = {}
    slide.close()

    top_statistics = aggregate_patch_group(top)

    middle_statistics = aggregate_patch_group(middle)

    bottom_statistics = aggregate_patch_group(bottom)
    # Metadata

    dataset = metadata.detect_dataset(slide_id)

    organ = metadata.lookup_organ(
        slide_id,
        manual_organ=manual_organ
    )

    # Build JSON

    slide_json = build_slide_json(
        slide_id=slide_id,
        dataset=dataset,
        organ=organ,
        ground_truth=CLASS_NAMES[label],
        prediction=prediction,
        top=top,
        middle=middle,
        bottom=bottom,
        top_statistics=top_statistics,
        middle_statistics=middle_statistics,
        bottom_statistics=bottom_statistics
    )

    # Save

    folder = create_slide_folder(output_dir, slide_id)

    save_json(
        slide_json,
        folder / "evidence.json"
    )

    return prediction.predicted_class


# Export all slides

def export_dataset(
    dataframe,
    dataset,
    model,
    metadata,
    output_dir,
    device,
    top_k,
    manual_organ=None
):

    predictions = []

    for sample in tqdm(dataset):

        pred = export_slide(
            sample=sample,
            model=model,
            metadata=metadata,
            output_dir=output_dir,
            top_k=top_k,
            device=device,
            manual_organ=manual_organ
        )

        predictions.append(pred)

    dataframe = dataframe.copy()
    dataframe["prediction"] = predictions

    dataframe.to_csv(
        Path(output_dir) / "predictions.csv",
        index=False
    )
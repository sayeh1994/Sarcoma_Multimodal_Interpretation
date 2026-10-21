# Sarcoma Multimodal Interpretation

An evidence-generation pipeline for multimodal interpretation of sarcoma whole-slide images (WSIs). The pipeline uses MIL model outputs, selected image patches, and cell-level information to support downstream analysis.

## Updates

- ✅ **Poster accepted at the ESMO AI & Digital Oncology Congress 2026.**
- 📝 **Training code:** The training script will be released after the congress.
- 📌 **MIL checkpoint access:** Please request access to the MIL model weights through the [access request form](https://forms.gle/ztvHVjnq46TVuzmd7).

> [!NOTE]
> This project uses [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) for cell detection. The pipeline requires a modified version that accepts coordinates for selected patches instead of processing an entire slide. The modifications are maintained separately and are not part of the original CellViT++ repository. Please access to the modified repository here: [modified CellViT repository](https://github.com/sayeh1994/CellViT-plus-plus-Patchwise.git).

## Installation

### Hardware requirements

- **CUDA-capable NVIDIA GPU**
- **Recommended VRAM:** 40 GB for inference; the minimum required VRAM may vary with the model, input size, and configuration.
- Experiments were run using **one NVIDIA A100 GPU with 40 GB VRAM**.

### 1. Create the Conda environment

```bash
conda env create -f environment.yaml
```

### 2. Activate the environment

```bash
conda activate sarcoma_mil
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch for your system

The environment was developed with the following PyTorch package versions:

- `torch==2.2.1`
- `torchaudio==2.2.1`
- `torchvision==0.17.1`

Choose a PyTorch build compatible with your operating system, Python version, and CUDA setup. Refer to the [official PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/) for the appropriate installation command.

> [!IMPORTANT]
> Even if PyTorch is installed as part of the Conda environment, verify that the installed build is compatible with your system. Install a specific build only when needed, and avoid mixing incompatible PyTorch, torchvision, torchaudio, and CUDA builds.

Example installation command (CUDA 12.1 build; check the official PyTorch instructions before using it):

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

## Running the pipeline

The examples below use `export_mil_features.py`.

For all examples, the input CSV should contain the following columns:

| Column | Description |
|---|---|
| `slide_id` | Identifier for the slide |
| `path` | Path to the MIL feature `.h5` file |
| `label` | Ground-truth or dataset label required by the pipeline |
| `wsi_path` | Path to the original WSI file (`.svs`) |

Use the metadata option appropriate for your dataset. Update all example paths before running the commands.

<details>
<summary><strong>1. Example using TCGA metadata</strong></summary>

```bash
python3 export_mil_features.py \
  --checkpoint /path/to/checkpoint.pt \
  --csv /path/to/input_data.csv \
  --output /path/to/output/ \
  --top_k 100 \
  --device cuda \
  --tcga_metadata /path/to/TCGA-SARC.GDC_phenotype.tsv
```

</details>

<details>
<summary><strong>2. Example using CPTAC metadata</strong></summary>

```bash
python3 export_mil_features.py \
  --checkpoint /path/to/checkpoint.pt \
  --csv /path/to/input_data.csv \
  --output /path/to/output/ \
  --top_k 100 \
  --device cuda \
  --cptac_metadata /path/to/TCIA_CPTAC_Pathology_Portal.csv
```

If your dataset requires manual organ mapping, add the following argument and replace the example value with the relevant organ:

```bash
--manual_organ "Uterus"
```

</details>

<details>
<summary><strong>3. Example using manual metadata</strong></summary>

```bash
python3 export_mil_features.py \
  --checkpoint /path/to/checkpoint.pt \
  --csv /path/to/input_data.csv \
  --output /path/to/output/ \
  --top_k 100 \
  --device cuda \
  --manual_organ "Uterus"
```

If multiple organ values are supported by the script, check its argument parser for the expected syntax before supplying more than one value.

</details>

## Notes

- Replace placeholder paths with paths valid on your system.
- The MIL checkpoint and modified CellViT code may require separate access approval.
- The exact hardware and memory requirements depend on the selected model and inference settings.
- The training script will be published after the congress.

## Citation

Citation information will be added when the associated work is available.

## License and access

Please check the repository license and the access conditions for the trained MIL weights and modified CellViT code before using or redistributing them.

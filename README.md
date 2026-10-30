# MMPL-Seg

Official research code for **MMPL-Seg: Prompt Learning with Missing Modalities for Medical Segmentation**, published at IEEE BIBM 2024.

MMPL-Seg addresses pulmonary infection segmentation when clinical text is partially or entirely unavailable. The model combines a UNet-style encoder-decoder, modality reconstruction and fusion, and dynamic convolution for image-text feature fusion.

- Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10822668)
- DOI: [10.1109/BIBM62325.2024.10822668](https://doi.org/10.1109/BIBM62325.2024.10822668)

## Repository structure

```text
MMPL-Seg/
├── Nets/
│   ├── Double_UnetModel4.py       # Main dynamic double-UNet model
│   ├── Double_UnetModel4_Ab*.py   # Ablation variants
│   ├── UNet.py                    # Image-only baseline
│   └── UNet_plus_bert.py          # Text-conditioned UNet baseline
├── Train_one_epoch.py             # Training and validation loop
├── data.py                        # QaTa-COV19-v2 data loaders
├── loss_poly.py                   # Segmentation and alignment losses
├── ramps.py                       # Semi-supervised ramp-up schedule
├── ssl_utils.py                   # Semi-supervised training helpers
├── train.py                       # Experiment entry point
├── test.py                        # Evaluation entry point
├── utils.py                       # Losses, metrics, and visualization helpers
└── requirements.txt
```

The ablation implementations are intentionally retained to support the experiments reported in the paper.

## Environment

Python 3.8 or later is recommended. Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your system if GPU training is required.

## Data preparation

The experiments use the [QaTa-COV19 dataset](https://www.kaggle.com/datasets/aysendegerli/qata-cov19-dataset). The dataset is not redistributed by this repository. By default, the scripts expect the following top-level layout:

```text
QaTa-COV19-v2/
├── Train_Set/
│   ├── Images/
│   ├── Ground-truths/
│   └── Text_Emb/
├── Val_Set/
│   ├── Images/
│   ├── Ground-truths/
│   └── Text_Emb/
└── Test_Set/                    # Optional evaluation split
```

Ground-truth filenames are expected to use the `mask_` prefix; the corresponding image filename is obtained by removing that prefix. Text embeddings are NumPy arrays with the same base filename as the mask. The main model expects embeddings with shape `[18, 256]`, including at least the first ten prompt tokens used by the fusion module. Update `train_data_path`, `val_data_path`, and `test_data_path` in the entry-point scripts when using a different location.

## Training and evaluation

The default `model_name` is `MMPLSeg`, which selects the main model in `Nets/Double_UnetModel4.py`. Adjust the experiment settings near the top of `train.py`, then run:

```bash
python3 train.py
```

Set the matching model and checkpoint path in `test.py`, then run. The default path is `params/Best_MMPLSeg.pth`; rename a selected checkpoint or update this path as needed.

```bash
python3 test.py
```

Checkpoints and datasets are excluded from version control. Create the relevant output directories before training if they do not already exist.

### Code snapshot note

The data loader, losses, and semi-supervised utilities used by the retained code are included. The experiment entry points also preserve references to optional third-party baselines from the original research workspace; those baseline implementations are not part of this code snapshot. Remove those optional imports or add the corresponding implementations before running an affected branch. The core model and ablation files are retained without changes to their computational logic.

## Citation

If this code is useful in your research, please cite:

```bibtex
@inproceedings{qu2024mmplseg,
  title     = {MMPL-Seg: Prompt Learning with Missing Modalities for Medical Segmentation},
  author    = {Qu, Ruowen and Wu, Wenxuan and Guan, Yeyi and Shu, Lin},
  booktitle = {2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages     = {5053--5058},
  year      = {2024},
  doi       = {10.1109/BIBM62325.2024.10822668}
}
```

## License

The source code is released under the [MIT License](LICENSE). Dataset files, pretrained weights, and third-party implementations remain subject to their respective licenses.

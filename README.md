# MMPL-Seg

Official research code for **MMPL-Seg: Prompt Learning with Missing Modalities for Medical Segmentation**, published at IEEE BIBM 2024.

MMPL-Seg is a text-guided medical image segmentation framework designed to remain robust when clinical text is partially or entirely unavailable. It combines modality reconstruction and fusion with dynamic convolution in a UNet-style architecture.

- [Paper](https://ieeexplore.ieee.org/document/10822668)
- [DOI](https://doi.org/10.1109/BIBM62325.2024.10822668)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your system if GPU training is required.

## Usage

The default configuration selects the main MMPL-Seg model in `Nets/Double_UnetModel4.py`. Experiment settings can be adjusted in `train.py` and `test.py`.

```bash
python3 train.py
python3 test.py
```

The `Nets` directory also contains the UNet baselines and ablation variants used in the experiments.

## Acknowledgements

This implementation references parts of the open-source [LViT codebase](https://github.com/HUANGLIZI/LViT), associated with [LViT: Language meets Vision Transformer in Medical Image Segmentation](https://arxiv.org/abs/2206.14718). We thank the authors for making their work publicly available.

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

This project is released under the [MIT License](LICENSE).

# AGF Node Classification

This directory contains the implementation of Approximate Graph Filter (AGF) for Node Classification on Cora, Citeseer, and Deezer datasets.

## Structure

- `data/`: Dataset loading utilities.
- `models/`: Model definitions.
  - `model.py`: Main `AGFNodeClassifier` model.
  - `layers.py`: Encoder and Attention layers.
  - `attention.py`: SVD, Softmax, and Hybrid attention mechanisms.
  - `polyconv.py`: Polynomial Graph Filter framework.
- `main.py`: Training and testing script.

## Usage

To train the model on Cora:

```bash
python main.py --dataset Cora --epochs 200 --attn_type svd --poly_type jacobi
```

To train on Citeseer:

```bash
python main.py --dataset Citeseer --epochs 200 --attn_type svd --poly_type jacobi
```

To train on Deezer (requires internet access to download dataset):

```bash
python main.py --dataset Deezer --epochs 200 --attn_type svd --poly_type jacobi
```

## Arguments

- `--dataset`: Dataset name (Cora, Citeseer, Deezer).
- `--d_model`: Hidden dimension size (default: 64).
- `--n_heads`: Number of attention heads (default: 4).
- `--num_layers`: Number of encoder layers (default: 2).
- `--lr`: Learning rate (default: 0.001).
- `--weight_decay`: Weight decay (default: 5e-4).
- `--epochs`: Number of training epochs (default: 200).
- `--reg_weight`: Weight for orthogonality loss (default: 0.1).
- `--attn_type`: Attention type (`svd`, `softmax`, `hybrid`).
- `--poly_type`: Polynomial filter type (`jacobi`, `chebyshev`, `monomial`, `legendre`).
- `--K`: Polynomial order (default: 3).
- `--device`: Device to use (`cuda` or `cpu`).

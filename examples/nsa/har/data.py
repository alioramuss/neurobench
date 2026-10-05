"""
NSA HAR data: WISDM smartwatch gyroscope, preprocessed and split by the NSA authors.

NSA publishes the exact train/test split it reports results on as four ``.npy`` files
(``x_train``, ``y_train``, ``x_test``, ``y_test``) inside ``WISDM.zip`` on Hugging Face.
Using that file, rather than re-running the preprocessing, keeps results comparable
with the NSA paper.

Shapes: ``x`` is ``(n_windows, 200, 3)`` float, ``y`` is ``(n_windows,)`` with 18 classes.
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import TensorDataset

WISDM_URL = (
    "https://huggingface.co/datasets/liyc5929/neuroseqbench/resolve/main/"
    "neuromorphic_sequential_arena/WISDM/WISDM.zip"
)
FILES = ("x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy")


def _find_split_dir(root):
    """Return the folder under ``root`` that holds all four split files, or None."""
    for dirpath, _, filenames in os.walk(root):
        if all(f in filenames for f in FILES):
            return dirpath
    return None


def download(root):
    """Download and unzip NSA's preprocessed WISDM split into ``root``."""
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "WISDM.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {WISDM_URL}")
        urllib.request.urlretrieve(WISDM_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)


def load_split(root, split, download_if_missing=True):
    """
    Load the NSA HAR train or test split as a ``TensorDataset``.

    Args:
        root: folder holding (or receiving) the WISDM ``.npy`` files.
        split: ``"train"`` or ``"test"``.
        download_if_missing: fetch ``WISDM.zip`` from Hugging Face if needed.

    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    split_dir = _find_split_dir(root)
    if split_dir is None:
        if not download_if_missing:
            raise FileNotFoundError(f"NSA WISDM files not found under {root}")
        download(root)
        split_dir = _find_split_dir(root)
        if split_dir is None:
            raise FileNotFoundError(f"WISDM.zip did not contain {FILES}")

    x = np.load(os.path.join(split_dir, f"x_{split}.npy"))
    y = np.load(os.path.join(split_dir, f"y_{split}.npy"))
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )

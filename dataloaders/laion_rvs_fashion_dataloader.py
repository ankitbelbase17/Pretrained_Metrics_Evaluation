"""
dataloaders/laion_rvs_fashion_dataloader.py
============================================
Dataset class for the LAION-RVS-Fashion dataset, loaded from HuggingFace Hub.

Streams examples from ``Slep/LAION-RVS-Fashion`` and materialises a local
buffer of ``limit`` samples so that standard integer indexing (__getitem__)
works with PyTorch DataLoader.

Output dict per sample (canonical format):
    person : Tensor (3, H, W) float32 [0, 1]
    cloth  : Tensor (3, H, W) float32 [0, 1]
    meta   : dict   {id, dataset}
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class LAIONRVSFashionDataset(Dataset):
    """
    Streaming-to-buffered loader for LAION-RVS-Fashion.

    Parameters
    ----------
    split : str
        HuggingFace split name (``"train"``).
    limit : int
        Maximum number of examples to buffer.
    img_size : tuple (H, W)
        Output image resolution.
    """

    def __init__(
        self,
        split: str = "train",
        limit: int = 1000,
        img_size: tuple = (512, 384),
        **kwargs,
    ):
        from datasets import load_dataset as hf_load_dataset

        self.transform = T.Compose([T.Resize(img_size), T.ToTensor()])
        self.data = []

        hf_ds = hf_load_dataset("Slep/LAION-RVS-Fashion", streaming=True)[split]
        it = iter(hf_ds)
        for _ in range(limit):
            try:
                self.data.append(next(it))
            except StopIteration:
                break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # Resolve image fields (HF datasets may use different column names)
        p_pil = item.get("person_image") or item.get("image")
        c_pil = item.get("cloth_image") or item.get("cloth")

        if p_pil is None:
            p_pil = Image.new("RGB", (512, 512), (128, 128, 128))
        if c_pil is None:
            c_pil = Image.new("RGB", (512, 512), (128, 128, 128))

        if not isinstance(p_pil, Image.Image):
            p_pil = Image.fromarray(np.array(p_pil))
        if not isinstance(c_pil, Image.Image):
            c_pil = Image.fromarray(np.array(c_pil))

        return {
            "person": self.transform(p_pil.convert("RGB")),
            "cloth": self.transform(c_pil.convert("RGB")),
            "meta": {"id": f"laion_{idx}", "dataset": "laion"},
        }


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset(
        "Slep/LAION-RVS-Fashion",
        streaming=True,
    )

    train = dataset["train"]

    for example in train:
        print(example)
        break

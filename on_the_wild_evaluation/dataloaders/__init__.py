"""
on_the_wild_evaluation/dataloaders/__init__.py
===============================================
Dataloaders for In-the-Wild Virtual Try-On Evaluation

Provides triplet dataloaders: (person_image, cloth_image, tryon_image)
"""

from .triplet_dataloader import (
    TripletDataLoader,
    TripletDataset,
    collate_triplets,
)

__all__ = [
    "TripletDataLoader",
    "TripletDataset",
    "collate_triplets",
]

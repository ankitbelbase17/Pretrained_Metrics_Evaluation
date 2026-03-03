import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ---------------------- CONFIGURATION ----------------------
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Define transforms for RGB images
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# For grayscale images like mask
basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

class VITONHDDataset(Dataset):
    def __init__(self, data_root_path, output_dir, eval_pair, split: str = "train"):
        self.data_root_path = data_root_path
        self.output_dir = output_dir
        self.eval_pair = eval_pair
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        # Resolve the split subdirectory (train / test)
        split_root = os.path.join(self.data_root_path, self.split)

        # Auto-detect mask folder: prefer "agnostic-mask", fall back to "agnostic"
        mask_folder = "agnostic-mask"
        if not os.path.exists(os.path.join(split_root, mask_folder)):
            mask_folder = "agnostic"

        # Build output dir path (used to skip already-generated images)
        output_dir = os.path.join(
            self.output_dir, "vitonhd",
            'unpaired' if not self.eval_pair else 'paired'
        )

        # Try to locate a pairs file (several common names)
        pair_txt = None
        for candidate in (
            'test_pairs.txt', 'train_pairs.txt',
            'test_pairs_unpaired.txt', 'train_pairs_unpaired.txt',
        ):
            p = os.path.join(self.data_root_path, candidate)
            if os.path.exists(p):
                pair_txt = p
                break

        data = []

        if pair_txt is not None:
            # ------------------------------------------------------------------
            # Pairs-file mode
            # ------------------------------------------------------------------
            with open(pair_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(" ")
                if len(parts) < 2:
                    continue
                person_img, cloth_img = parts[0], parts[1]
                if os.path.exists(os.path.join(output_dir, person_img)):
                    continue
                if self.eval_pair:
                    cloth_img = person_img
                mask_name = (
                    person_img.replace('.jpg', '_mask.png')
                    if mask_folder == 'agnostic-mask'
                    else person_img
                )
                data.append({
                    'person_name': person_img,
                    'person': os.path.join(split_root, 'image', person_img),
                    'cloth':  os.path.join(split_root, 'cloth', cloth_img),
                    'mask':   os.path.join(split_root, mask_folder, mask_name),
                })
        else:
            # ------------------------------------------------------------------
            # Fallback: scan the image directory and create pairs on the fly
            # ------------------------------------------------------------------
            image_dir = os.path.join(split_root, 'image')
            if not os.path.exists(image_dir):
                return data
            all_images = sorted(
                f for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            for i, person_img in enumerate(all_images):
                if os.path.exists(os.path.join(output_dir, person_img)):
                    continue
                # Paired: cloth == person.  Unpaired: rotate to next image.
                cloth_img = (
                    person_img
                    if self.eval_pair
                    else all_images[(i + 1) % len(all_images)]
                )
                mask_name = (
                    person_img.replace('.jpg', '_mask.png')
                    if mask_folder == 'agnostic-mask'
                    else person_img
                )
                data.append({
                    'person_name': person_img,
                    'person': os.path.join(split_root, 'image', person_img),
                    'cloth':  os.path.join(split_root, 'cloth', cloth_img),
                    'mask':   os.path.join(split_root, mask_folder, mask_name),
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        # Load images
        person_image_pil = Image.open(data['person']).convert("RGB")
        cloth_image = Image.open(data['cloth']).convert("RGB")
        mask_pil = Image.open(data['mask']).convert("L")

        # ---------------------------
        # Generate overlay image
        # ---------------------------
        person_np = np.array(person_image_pil)
        mask_np = np.array(mask_pil)

        # Blur mask
        blurred_mask = cv2.GaussianBlur(mask_np, (21, 21), sigmaX=10)
        alpha_mask = blurred_mask.astype(np.float32) / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)  # Shape: (H, W, 1)

        # Create grey overlay
        grey_overlay = np.full_like(person_np, fill_value=128, dtype=np.uint8)

        # Blend images
        overlay_np = (person_np * (1 - alpha_mask) + grey_overlay * alpha_mask).astype(np.uint8)

        # Convert back to PIL and transform to tensor
        overlay_pil = Image.fromarray(overlay_np)
        overlay_image = transform(overlay_pil)

        # Transform other inputs
        person_image = transform(person_image_pil)
        cloth_image = transform(cloth_image)
        mask = basic_transform(mask_pil).squeeze(0)  # [H, W]

        # Extract base filename
        base_name = os.path.splitext(data['person_name'])[0]

        return {
            'person_image': person_image,
            'cloth_image': cloth_image,
            'mask': mask,
            'overlay_image': overlay_image,
            'filename': base_name
        }


def custom_collate_fn(batch):
    """
    Collate function that:
      - Applies Gaussian blur to mask edges
      - Batches the precomputed overlay_image from each sample
    """
    person_images   = []
    cloth_images    = []
    masks           = []
    overlay_images  = []
    filenames       = []

    for item in batch:
        # 1) Blur the raw mask:
        mask_np = item['mask'].cpu().numpy().astype(np.uint8)  # [H, W]
        blurred_mask = cv2.GaussianBlur(mask_np, (39, 39), 0)  # still [H, W]
        blurred_mask = blurred_mask.astype(np.float32) / 255.0
        blurred_mask_tensor = torch.from_numpy(blurred_mask)

        # 2) Collect everything else:
        person_images.append(   item['person_image'] )   # [C, H, W]
        cloth_images.append(    item['cloth_image'] )    # [C, H, W]
        masks.append(           blurred_mask_tensor )    # [H, W]
        overlay_images.append(  item['overlay_image'] )  # [C, H, W]
        filenames.append(       item['filename'] )

    batch_dict = {
        'person_image':  torch.stack(person_images),
        'cloth_image':   torch.stack(cloth_images),
        'mask':          torch.stack(masks),
        'overlay_image': torch.stack(overlay_images),
        'filename':      filenames
    }
    return batch_dict


if __name__ == "__main__":
    dataset = VITONHDDataset(
        data_root_path="benchmark_datasets/viton_hd",
        output_dir="output",
        eval_pair=True,
        split="test",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Test one batch:
    for batch in dataloader:
        print("Person:",   batch["person_image"].shape)
        print("Cloth:",    batch["cloth_image"].shape)
        print("Mask:",     batch["mask"].shape)
        print("Overlay:",  batch["overlay_image"].shape)
        print("Filenames:", batch["filename"])
        break

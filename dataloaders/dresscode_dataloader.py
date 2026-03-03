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

# For grayscale images like depth and mask
basic_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

class Dresscode(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory containing subfolders: upper_body, lower_body, dresses
        """
        self.root_dir = root_dir
        self.categories = ['upper_body', 'lower_body', 'dresses']
        self.data = self.load_data()

    def load_data(self):
        """Load data from all category folders"""
        data = []
        for category in self.categories:
            category_path = os.path.join(self.root_dir, category)
            
            # Define subdirectories for this category
            person_dir = os.path.join(category_path, "image")
            cloth_dir = os.path.join(category_path, "cloth")
            normal_dir = os.path.join(category_path, "normal")
            depth_dir = os.path.join(category_path, "depth")
            mask_dir = os.path.join(category_path, "mask")
            caption_dir = os.path.join(category_path, "caption")
            
            # Skip if category doesn't exist
            if not os.path.exists(person_dir):
                continue
            
            # Get list of image files in this category
            image_files = [
                f for f in os.listdir(person_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            # Add all images from this category
            for img_file in image_files:
                base_name, _ = os.path.splitext(img_file)
                data.append({
                    'category': category,
                    'img_file': img_file,
                    'base_name': base_name,
                    'person_path': os.path.join(person_dir, img_file),
                    'cloth_path': os.path.join(cloth_dir, img_file),
                    'normal_path': os.path.join(normal_dir, f"{base_name}.jpg"),
                    'depth_path': os.path.join(depth_dir, f"{base_name}.jpg"),
                    'mask_path': os.path.join(mask_dir, f"{base_name}.png"),
                    'caption_path': os.path.join(caption_dir, f"{base_name}.txt")
                })
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data entry
        item = self.data[idx]
        
        # Build file paths from stored paths
        person_path = item['person_path']
        cloth_path = item['cloth_path']
        normal_path = item['normal_path']
        depth_path = item['depth_path']
        mask_path = item['mask_path']
        caption_path = item['caption_path']

        # Load images
        person_image_pil = Image.open(person_path).convert("RGB")
        cloth_image = Image.open(cloth_path).convert("RGB")
        normal_map = Image.open(normal_path).convert("RGB")
        depth_map = Image.open(depth_path).convert("L")
        mask_pil = Image.open(mask_path).convert("L")

        # ---------------------------
        # Generate overlay image here
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
        overlay_image = transform(overlay_pil)  # Same transform as person_image

        # Transform other inputs
        person_image = transform(person_image_pil)
        cloth_image = transform(cloth_image)
        normal_map = transform(normal_map)
        depth_map = basic_transform(depth_map)     # [1, H, W]
        mask = basic_transform(mask_pil).squeeze(0)  # [H, W]

        # Load caption (optional)
        caption = ""
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except FileNotFoundError:
            pass

        return {
            'person_image': person_image,
            'cloth_image': cloth_image,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'mask': mask,
            'overlay_image': overlay_image,  # <<<< NEW ENTRY
            'caption': caption,
            'filename': item['base_name']
        }

def custom_collate_fn(batch):
    """
    Collate function that:
      - Applies Gaussian blur to mask edges
      - Returns captions only 20% of the time
      - Batches the precomputed overlay_image from each sample
    """
    include_caption = (random.random() < 0.2)

    person_images   = []
    cloth_images    = []
    normal_maps     = []
    depth_maps      = []
    masks           = []
    overlay_images  = []
    captions        = []
    filenames       = []

    for item in batch:
        # 1) Blur the raw mask:
        mask_np = item['mask'].cpu().numpy().astype(np.uint8)  # [H, W]
        blurred_mask = cv2.GaussianBlur(mask_np, (39, 39), 0)  # still [H, W]
        blurred_mask = blurred_mask.astype(np.float32) / 255.0
        # if you want a channel dimension (1×H×W), uncomment next line:
        # blurred_mask = np.expand_dims(blurred_mask, axis=0)
        blurred_mask_tensor = torch.from_numpy(blurred_mask)

        # 2) Collect everything else:
        person_images.append(   item['person_image'] )   # [C, H, W]
        cloth_images.append(    item['cloth_image'] )    # [C, H, W]
        normal_maps.append(     item['normal_map'] )     # [C, H, W]
        depth_maps.append(      item['depth_map'] )      # [1, H, W]
        masks.append(           blurred_mask_tensor )    # [H, W] or [1, H, W]
        overlay_images.append(  item['overlay_image'] )  # [C, H, W]
        captions.append(        item['caption'] if include_caption else "" )
        filenames.append(       item['filename'] )

    batch_dict = {
        'person_image':  torch.stack(person_images),
        'cloth_image':   torch.stack(cloth_images),
        'normal_map':    torch.stack(normal_maps),
        'depth_map':     torch.stack(depth_maps),
        'mask':          torch.stack(masks),
        'overlay_image': torch.stack(overlay_images),
        'caption':       captions,
        'filename':      filenames
    }
    return batch_dict


if __name__ == "__main__":
    dataset = Dresscode(root_dir="benchmark_datasets/dresscode")
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
        print("Normal:",   batch["normal_map"].shape)
        print("Depth:",    batch["depth_map"].shape)
        print("Mask:",     batch["mask"].shape)
        print("Overlay:",  batch["overlay_image"].shape)
        print("Captions:", batch["caption"])
        break

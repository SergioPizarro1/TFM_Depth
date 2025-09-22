# datasets/nyu_depth_v2.py
import os, random
from typing import Tuple, Callable, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

class PairedRandomHorizontalFlip:
    """ Flip horizontal conjunto RGB/Depth con probabilidad p."""
    def __init__(self, p=0.5): 
        self.p = p
    def __call__(self, img, depth):
        if random.random() < self.p:
            img   = TF.hflip(img)
            depth = TF.hflip(depth)
        return img, depth

class PairedColorJitter:
    """ Jitter conjunto RGB/Depth (sólo afecta a RGB)."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.brightness = brightness 
        self.contrast=contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, img, depth):
        # sólo a la imagen RGB
        img = TF.adjust_brightness(img, 1.0 + (2*random.random()-1.0)*self.brightness)
        img = TF.adjust_contrast(img,   1.0 + (2*random.random()-1.0)*self.contrast)
        img = TF.adjust_saturation(img, 1.0 + (2*random.random()-1.0)*self.saturation)
        img = TF.adjust_hue(img,        (2*random.random()-1.0)*self.hue)
        return img, depth

class PairRandomRotate:
    """
    Rotación conjunta RGB/Depth en grados pequeños (p.ej. +/-5).
    - fill RGB: negro
    - fill Depth: 0 (se considera inválido y la máscara lo filtrará)
    - RGB: bilinear, Depth: nearest
    """
    def __init__(self, degrees=2.5, p=0.5):
        self.degrees = float(degrees)
        self.p = float(p)

    def __call__(self, img, depth):
        if random.random() > self.p or self.degrees <= 0:
            return img, depth
        angle = random.uniform(-self.degrees, self.degrees)
        img_r   = TF.rotate(img,   angle, interpolation=TF.InterpolationMode.BILINEAR, fill=(0,0,0))
        depth_r = TF.rotate(depth, angle, interpolation=TF.InterpolationMode.NEAREST,  fill=0)
        return img_r, depth_r
""""
class PairedRandomResizedCrop:
    #Recorte y *resize* idénticos para img y depth.
    def __init__(self, size: Tuple[int,int], scale=(0.8, 1.0), ratio=(1.0,1.0)):
        self.size = size
        self.scale=scale
        self.ratio=ratio
    def __call__(self, img, depth):
        i, j, h, w =  TF.get_params(img, self.scale, self.ratio)
        img   = TF.resized_crop(img,   i, j, h, w, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.resized_crop(depth, i, j, h, w, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return img, depth
"""

# datasets/nyu_depth_v2_kaggle.py
class NYUDepthV2CSV(Dataset):
    def __init__(self, csv_file,
                 image_size=(480,640),
                 max_depth=10.0,
                 augmentation=True,
                 mean=(0.485,0.456,0.406),
                 std=(0.229,0.224,0.225)):
        self.df = pd.read_csv(csv_file)
        self.size = image_size
        self.max_depth = max_depth
        self.aug = augmentation
        self.mean = mean
        self.std = std
        #print(self.df.head())

        # augmentations compartidas 
        self.paired = []
        if self.aug:
            self.paired += [
                PairedRandomHorizontalFlip(p=0.5),
                #PairedRandomResizedCrop(size=image_size, scale=(0.9,1.0), ratio=(1.0,1.0)),
                PairedColorJitter(0.2,0.2,0.2,0.02),
                PairRandomRotate(degrees=2.5, p=0.5)
            ]

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        rgb_path   = self.df.iloc[idx, 0]
        depth_path = self.df.iloc[idx, 1]
        #print("RGB path:", rgb_path, "Depth path:", depth_path)

        img = Image.open("Data/nyu_data/"+rgb_path).convert("RGB")
        depth_img = Image.open("Data/nyu_data/" + depth_path)
        depth_np  = np.array(depth_img)

        # --- Convertir SIEMPRE a METROS con lógica según tipo ---
        if depth_np.dtype == np.uint16 or depth_img.mode in ("I;16", "I"):
            # Profundidad en milímetros -> metros
            depth_m = depth_np.astype(np.float32) / 1000.0
        elif depth_np.dtype == np.uint8 or depth_img.mode == "L":
            # Profundidad normalizada 8-bit (0..255) -> [0, max_depth] m
            depth_m = depth_np.astype(np.float32) * (self.max_depth / 255.0)
        elif depth_img.mode == "F" or depth_np.dtype in (np.float32, np.float64):
            # Ya viene en metros
            depth_m = depth_np.astype(np.float32)
        else:
            # Fallback
            depth_m = depth_np.astype(np.float32)

        # Clip de seguridad a [0, max_depth]
        depth_m = np.clip(depth_m, 0.0, self.max_depth)

        # Volvemos a PIL 'F' (float32) para mantener el pipeline de augmentations emparejado
        depth = Image.fromarray(depth_m)    

        for t in self.paired:
            img, depth = t(img, depth)

        img_t = TF.to_tensor(img)
        depth_t = torch.from_numpy(np.array(depth).astype(np.float32)).unsqueeze(0)
        img_t = TF.normalize(img_t, self.mean, self.std)

        H, W = self.size
        if (img_t.shape[-2], img_t.shape[-1]) != (H, W):
            img_t = TF.resize(img_t, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            depth_t = TF.resize(depth_t, [H, W], interpolation=TF.InterpolationMode.NEAREST)

        focal = 518.8579
        return img_t, depth_t, focal


def make_nyu_csv_loaders(
    train_csv: str = "./Data/nyu_data/data/nyu2_train.csv",
    val_csv:   str = "./Data/nyu_data/data/nyu2_test.csv",
    image_size: Tuple[int, int] = (480, 640),
    batch_size: int = 8,
    num_workers: int = 4,
    max_depth: float = 10.0,
    augment_train: bool = True,
):
    """
    Crea DataLoaders para entrenamiento y validación usando los CSV.
    Ajustar train_csv/val_csv si cambia la estructura de carpetas.
    """
    train_set = NYUDepthV2CSV(
        csv_file=train_csv,
        image_size=image_size,
        max_depth=max_depth,
        augmentation=augment_train
    )
    val_set = NYUDepthV2CSV(
        csv_file=val_csv,
        image_size=image_size,
        max_depth=max_depth,
        augmentation=False
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ---------------------------- #
#  Sanity check
# ---------------------------- #

if __name__ == "__main__":
    # Ajusta si mueves los CSV
    tl, vl = make_nyu_csv_loaders(
        image_size=(480, 640),
        batch_size=4,
        num_workers=0,
        max_depth=10.0,
        augment_train=True
    )
    print(len(tl), len(vl))
    xb, yb = next(iter(tl))
    print("xb:", xb.shape, xb.dtype, xb.min().item(), xb.max().item())
    print("yb:", yb.shape, yb.dtype, yb.min().item(), yb.max().item())
    # Esperado:
    # xb -> torch.Size([2, 3, 480, 640]) float32 (normalizado)
    # yb -> torch.Size([2, 1, 480, 640]) float32 en [0, 10]

"""
============================================================================
  ISIC 2024 Dataset — Data loading, transforms, and stratified splitting.

  Supports:
    • Individual image files (JPG)
    • HDF5 archive (train-image.hdf5)
    • Auto-detection of dataset paths (local + Kaggle)
    • Patient-level stratified splitting
    • Benign subsampling for class balance
============================================================================
"""

import os
import io
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import config as cfg


# ── Dataset ────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """
    PyTorch Dataset for the ISIC 2024 Skin Cancer challenge.
    Loads images from individual files OR from an HDF5 archive.
    """

    def __init__(self, dataframe, image_dir, transform=None, hdf5_path=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self._hdf5_file = None
        self._error_printed = False

    def _get_hdf5(self):
        """Lazy-open HDF5 file (one handle per worker process)."""
        if self._hdf5_file is None and self.hdf5_path and HAS_H5PY:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]["isic_id"]
        image = None

        # Try 1: Individual image file
        if self.image_dir:
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    if not self._error_printed:
                        print(f"\n⚠️ Error loading image file {img_path}: {e}")
                        self._error_printed = True

        # Try 2: HDF5 archive
        if image is None and self.hdf5_path and HAS_H5PY:
            try:
                hf = self._get_hdf5()
                if isic_id in hf:
                    img_bytes = hf[isic_id][()]
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                if not self._error_printed:
                    print(f"\n⚠️ Error loading image {isic_id} from HDF5: {e}")
                    self._error_printed = True

        # Fallback: black placeholder
        if image is None:
            image = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), color="black")

        target = self.data_frame.iloc[idx]["target"]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)


# ── Transforms ─────────────────────────────────────────────────────────

def get_train_transform(image_size=None):
    sz = image_size or cfg.IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_test_transform(image_size=None):
    sz = image_size or cfg.IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Dataset discovery ─────────────────────────────────────────────────

def _try_find_dataset(base_dir):
    """Search for train-metadata.csv and return (csv_path, image_dir) or (None, None)."""
    if not os.path.isdir(base_dir):
        return None, None

    # Strategy 1: base_dir is the dataset root
    candidate_csv = os.path.join(base_dir, "train-metadata.csv")
    if os.path.exists(candidate_csv):
        for img_sub in ["train-image/image", "train-image", "train-images/image", "train-images"]:
            candidate_img = os.path.join(base_dir, img_sub)
            if os.path.isdir(candidate_img):
                print(f"✅ Found ISIC dataset at: {base_dir}/")
                return candidate_csv, candidate_img
        candidate_img = os.path.join(base_dir, "train-image")
        print(f"✅ Found ISIC dataset at: {base_dir}/")
        return candidate_csv, candidate_img

    # Strategy 2: one-level-deep subdirectories
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        candidate_csv = os.path.join(folder_path, "train-metadata.csv")
        if os.path.exists(candidate_csv):
            img_dir = None
            for img_sub in ["train-image/image", "train-image", "train-images/image", "train-images"]:
                candidate_img = os.path.join(folder_path, img_sub)
                if os.path.isdir(candidate_img):
                    img_dir = candidate_img
                    break
            if img_dir is None:
                img_dir = os.path.join(folder_path, "train-image")
            print(f"✅ Found ISIC dataset at: {folder_path}/")
            return candidate_csv, img_dir

    return None, None


# ── DataLoader factory ─────────────────────────────────────────────────

def get_isic_dataloaders(
    batch_size=None,
    test_ratio=None,
    subsample_ratio=None,
):
    """
    Returns (train_loader, test_loader, num_classes, class_weights).

    Uses patient-level stratified 80/20 split.
    Falls back to dummy data if the dataset is not found.
    """
    batch_size = batch_size or cfg.BATCH_SIZE
    test_ratio = test_ratio or cfg.TEST_RATIO
    subsample_ratio = subsample_ratio if subsample_ratio is not None else cfg.SUBSAMPLE_RATIO
    num_classes = cfg.NUM_CLASSES

    # ── Locate dataset ──
    csv_path, image_dir = None, None
    for search_dir in cfg.DATA_SEARCH_DIRS:
        csv_path, image_dir = _try_find_dataset(search_dir)
        if csv_path is not None:
            break

    # Kaggle deep scan fallback
    if csv_path is None and os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root.replace("/kaggle/input", "").count(os.sep)
            if depth > 2:
                continue
            if "train-metadata.csv" in files:
                csv_path = os.path.join(root, "train-metadata.csv")
                for img_sub in ["train-image/image", "train-image", "train-images/image", "train-images"]:
                    candidate_img = os.path.join(root, img_sub)
                    if os.path.isdir(candidate_img):
                        image_dir = candidate_img
                        break
                if image_dir is None:
                    image_dir = os.path.join(root, "train-image")
                print(f"✅ Found ISIC dataset via deep scan at: {root}/")
                break

    if csv_path:
        print(f"📂 CSV path:   {csv_path}")
        print(f"📂 Image dir:  {image_dir}")
    else:
        print("❌ train-metadata.csv not found in any search directories.")

    train_tf = get_train_transform()
    test_tf  = get_test_transform()

    # ── Dummy fallback ──
    if csv_path is None or not os.path.exists(csv_path):
        print("⚠ ISIC dataset not found. Falling back to dummy data.")
        X = torch.randn(200, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, Y)
        tr = Subset(full, range(160))
        te = Subset(full, range(160, 200))
        return (
            DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=batch_size),
            num_classes,
            torch.ones(num_classes),
        )

    # ── Load CSV ──
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded CSV with {len(df)} rows, columns: {list(df.columns[:5])}")

    # Detect HDF5 archive
    hdf5_path = None
    dataset_root = os.path.dirname(csv_path)
    for hdf5_name in ["train-image.hdf5", "train-image.h5"]:
        candidate = os.path.join(dataset_root, hdf5_name)
        if os.path.exists(candidate):
            hdf5_path = candidate
            print(f"📂 HDF5 archive: {hdf5_path}")
            break

    if os.path.isdir(dataset_root):
        print(f"📂 Dataset contents: {os.listdir(dataset_root)}")

    # ── Patient-level stratified split ──
    if "patient_id" in df.columns:
        df = df.dropna(subset=["patient_id"]).reset_index(drop=True)
        patient_df = df.groupby("patient_id")["target"].max().reset_index()
        train_patients, test_patients = train_test_split(
            patient_df, test_size=test_ratio, stratify=patient_df["target"], random_state=42,
        )
        train_df = df[df["patient_id"].isin(train_patients["patient_id"])].reset_index(drop=True)
        test_df  = df[df["patient_id"].isin(test_patients["patient_id"])].reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, stratify=df["target"], random_state=42,
        )

    # ── Subsample benign class ──
    if subsample_ratio is not None and subsample_ratio > 0:
        train_malignant = train_df[train_df["target"] == 1]
        train_benign    = train_df[train_df["target"] == 0]
        n_mal = len(train_malignant)
        if n_mal > 0:
            n_benign = min(len(train_benign), n_mal * subsample_ratio)
            train_benign_sampled = train_benign.sample(n=n_benign, random_state=42)
            train_df = pd.concat([train_malignant, train_benign_sampled]).reset_index(drop=True)
            train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f"📉 Subsampled training set: {n_mal} malignant, {len(train_benign_sampled)} benign.")

    print(f"📊 Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")

    train_ds = ISICDataset(train_df, image_dir, transform=train_tf, hdf5_path=hdf5_path)
    test_ds  = ISICDataset(test_df,  image_dir, transform=test_tf,  hdf5_path=hdf5_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, prefetch_factor=2)

    # ── Class weights (dampened inverse frequency) ──
    class_counts = train_df["target"].value_counts().sort_index()
    total = len(train_df)
    cw_raw = [math.sqrt(total / class_counts.get(c, 1)) for c in range(num_classes)]
    majority_weight = cw_raw[0]
    cw = torch.tensor([w / majority_weight for w in cw_raw], dtype=torch.float32)
    print(f"⚖️  Class weights: {dict(enumerate(cw.tolist()))}")

    return train_loader, test_loader, num_classes, cw

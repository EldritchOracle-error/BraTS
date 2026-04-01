"""
save_preprocessed_2020.py

Preprocesses the BraTS 2020 H5 dataset.

Each .h5 file is already one 2D axial slice containing:
    'image'  → shape (240, 240, 4)  float64  — 4 modalities (flair, t1, t1ce, t2)
    'mask'   → shape (240, 240, 3)  uint8    — one-hot across 3 tumour classes

One-hot mask channel mapping:
    channel 0 → necrotic core        → label 1
    channel 1 → peritumoral edema    → label 2
    channel 2 → enhancing tumour     → label 3
    all zeros → background           → label 0

Output per non-empty slice:
    brats20_<stem>_img.npy   → shape (4, 128, 128)  float32
    brats20_<stem>_mask.npy  → shape (128, 128)     int64

Run this script ONCE before training. It only takes a few minutes
because the slicing is already done by the dataset provider.
"""

import os
import numpy as np
import h5py
from scipy.ndimage import zoom
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH = r"D:\AI_in_Organ_Design\datasets\BraTS_2020\BraTS2020_training_data\content\data"
SAVE_PATH = r"D:\AI_in_Organ_Design\datasets\BraTS_combined\preprocessed"
IMG_SIZE  = 128
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_PATH, exist_ok=True)


def normalize(volume):
    """Min-max normalise a single-channel 2D array to [0, 1]."""
    min_val = volume.min()
    max_val = volume.max()
    if max_val - min_val == 0:
        return volume.astype(np.float32)
    return ((volume - min_val) / (max_val - min_val)).astype(np.float32)


def resize_2d(arr, target=128, order=1):
    """Resize a 2D array to target×target using scipy zoom."""
    h, w = arr.shape
    return zoom(arr, (target / h, target / w), order=order)


def onehot_to_label(mask_onehot):
    """
    Convert a (H, W, 3) one-hot mask to a (H, W) integer label map.

    Channel assignment:
        0 → necrotic core     → label 1
        1 → peritumoral edema → label 2
        2 → enhancing tumour  → label 3
        background (no channel active) → label 0
    """
    label_map = np.zeros(mask_onehot.shape[:2], dtype=np.int64)
    # iterate in reverse priority so that in the rare case of overlap
    # the lower-index (higher-priority) channel wins
    for ch, label in [(2, 3), (1, 2), (0, 1)]:
        label_map[mask_onehot[:, :, ch] > 0] = label
    return label_map


# ─── GATHER FILES ─────────────────────────────────────────────────────────────
h5_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".h5")])

print(f"Found {len(h5_files)} .h5 slice files")
print(f"Saving to: {SAVE_PATH}")
print("=" * 60)

saved_count   = 0
skipped_empty = 0
skipped_error = 0

for fname in tqdm(h5_files, desc="Processing BraTS-2020 slices"):
    try:
        fpath = os.path.join(DATA_PATH, fname)

        with h5py.File(fpath, "r") as f:
            image = f["image"][()]   # (240, 240, 4)  float64
            mask  = f["mask"][()]    # (240, 240, 3)  uint8

        # Convert one-hot mask → integer label map (240, 240)
        label_map = onehot_to_label(mask)

        # Skip slices with no tumour annotation at all
        if label_map.max() == 0:
            skipped_empty += 1
            continue

        # Normalise each modality independently
        channels = []
        for c in range(image.shape[2]):          # 4 modalities
            channels.append(normalize(image[:, :, c]))

        # Resize each channel and the label map to IMG_SIZE × IMG_SIZE
        channels_resized = [resize_2d(ch, IMG_SIZE, order=1) for ch in channels]
        label_resized    = resize_2d(label_map.astype(np.float32), IMG_SIZE, order=0)

        # Stack channels → (4, 128, 128) and round label back to int
        img_slice  = np.stack(channels_resized, axis=0).astype(np.float32)
        mask_slice = np.round(label_resized).astype(np.int64)
        mask_slice = np.clip(mask_slice, 0, 3)

        # Use the h5 filename stem as the identifier
        # e.g. volume_100_slice_0.h5 → brats20_volume_100_slice_0
        stem   = os.path.splitext(fname)[0]          # "volume_100_slice_0"
        prefix = f"brats20_{stem}"

        np.save(os.path.join(SAVE_PATH, f"{prefix}_img.npy"),  img_slice)
        np.save(os.path.join(SAVE_PATH, f"{prefix}_mask.npy"), mask_slice)
        saved_count += 1

    except Exception as e:
        print(f"\nSkipping {fname}: {e}")
        skipped_error += 1

print(f"\nDone!")
print(f"  Slices saved        : {saved_count}")
print(f"  Skipped (empty)     : {skipped_empty}")
print(f"  Skipped (error)     : {skipped_error}")
print(f"  Output folder       : {SAVE_PATH}")
print()
print("Next steps:")
print("  1. python copy_2021_to_combined.py   ← merge BraTS-2021 slices in")
print("  2. python Train_Unet.py              ← train on the combined dataset")

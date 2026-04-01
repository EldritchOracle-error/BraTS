"""
verify_mask_channels.py

Scans the first 500 .h5 files to find a non-empty mask, then prints
per-channel statistics so we can confirm the channel→label mapping.

Run before save_preprocessed_2020.py to validate the mask interpretation.
"""

import os
import h5py
import numpy as np

DATA_PATH = r"D:\AI_in_Organ_Design\datasets\BraTS_2020\BraTS2020_training_data\content\data"

h5_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".h5")])
print(f"Scanning up to 500 files to find a non-empty mask...")

found = 0
for fname in h5_files[:500]:
    with h5py.File(os.path.join(DATA_PATH, fname), "r") as f:
        mask = f["mask"][()]    # (240, 240, 3)
        image = f["image"][()]  # (240, 240, 4)

    if mask.max() == 0:
        continue

    found += 1
    print(f"\nFound non-empty mask in: {fname}")
    print(f"  image shape : {image.shape}  dtype: {image.dtype}")
    print(f"  mask shape  : {mask.shape}   dtype: {mask.dtype}")
    print(f"  image range : min={image.min():.4f}  max={image.max():.4f}")
    print()
    print("  Mask channel statistics:")
    for ch in range(mask.shape[2]):
        ch_data = mask[:, :, ch]
        nonzero = np.count_nonzero(ch_data)
        unique  = sorted(np.unique(ch_data).tolist())
        print(f"    channel {ch}: nonzero pixels={nonzero:5d}  unique values={unique}")
    print()
    print("  Combined mask (any channel active):")
    combined = mask.any(axis=2)
    print(f"    total tumour pixels: {combined.sum()}")
    print()

    # Show what the converted label map looks like
    label_map = np.zeros(mask.shape[:2], dtype=np.int64)
    for ch, label in [(2, 3), (1, 2), (0, 1)]:
        label_map[mask[:, :, ch] > 0] = label
    print("  Converted label map unique values:", sorted(np.unique(label_map).tolist()))
    print("  Expected: subset of [0, 1, 2, 3]")

    if found >= 3:   # check 3 non-empty slices then stop
        break

if found == 0:
    print("No non-empty masks found in the first 500 files.")
    print("Try increasing the scan range or check the DATA_PATH.")

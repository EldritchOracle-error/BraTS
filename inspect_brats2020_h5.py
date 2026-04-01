"""
inspect_brats2020_h5.py

Run this ONCE to print the internal structure of the first .h5 file
so we know the exact key names, shapes, and dtypes to use in preprocessing.
"""

import os
import h5py

DATA_PATH = r"D:\AI_in_Organ_Design\datasets\BraTS_2020\BraTS2020_training_data\content\data"

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Path not found: {DATA_PATH}")
    exit()

h5_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".h5")])
print(f"Found {len(h5_files)} .h5 files in {DATA_PATH}")
print()

if not h5_files:
    print("No .h5 files found. Check DATA_PATH.")
    exit()

# Inspect the first file in detail
sample_path = os.path.join(DATA_PATH, h5_files[0])
print(f"Inspecting: {h5_files[0]}")
print("=" * 60)

with h5py.File(sample_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  DATASET  '{name}'")
            print(f"           shape : {obj.shape}")
            print(f"           dtype : {obj.dtype}")
            print(f"           min   : {obj[()].min():.4f}  max: {obj[()].max():.4f}")
            import numpy as np
            if "seg" in name.lower() or "mask" in name.lower() or "label" in name.lower():
                print(f"           unique values: {sorted(set(obj[()].flatten().tolist()))}")
        elif isinstance(obj, h5py.Group):
            print(f"  GROUP    '{name}'")

    f.visititems(print_structure)

print()
print("Top-level keys:", list(h5py.File(sample_path, "r").keys()))
print()
print("First 5 filenames:")
for name in h5_files[:5]:
    print(f"  {name}")

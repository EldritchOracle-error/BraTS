"""
save_preprocessed_2020.py

Preprocesses the BraTS 2020 Training dataset and saves each non-empty
axial slice as a pair of .npy files:
    <patient_id>_slice<NNN>_img.npy   → shape (4, 128, 128) float32
    <patient_id>_slice<NNN>_mask.npy  → shape (128, 128)    int64

BraTS 2020 naming convention:
    BraTS20_Training_001/
        BraTS20_Training_001_flair.nii.gz
        BraTS20_Training_001_t1.nii.gz
        BraTS20_Training_001_t1ce.nii.gz
        BraTS20_Training_001_t2.nii.gz
        BraTS20_Training_001_seg.nii.gz   ← only in training split (369 patients)

Label mapping (same as BraTS 2021):
    0 → background
    1 → necrotic core (NCR/NET)
    2 → peritumoral edema (ED)
    4 → enhancing tumor (ET)  ← remapped to 3

Run this script ONCE before training. It takes ~10-20 minutes.
"""

import os
import numpy as np
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-6
from scipy.ndimage import zoom
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_PATH = r"D:\\AI_in_Organ_Design\\datasets\\BraTS_2020\\extracted"
SAVE_PATH = r"D:\\AI_in_Organ_Design\\datasets\\BraTS_combined\\preprocessed"
IMG_SIZE  = 128
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_PATH, exist_ok=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def normalize(volume):
    """Min-max normalise to [0, 1]. Returns unchanged volume if flat."""
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val == 0:
        return volume
    return (volume - min_val) / (max_val - min_val)


def resize_volume(volume, target_size=128):
    """Resize H×W to target_size×target_size, preserving depth axis."""
    h, w = volume.shape[:2]
    scale_h = target_size / h
    scale_w = target_size / w
    if volume.ndim == 3:
        return zoom(volume, (scale_h, scale_w, 1), order=1)
    else:
        return zoom(volume, (scale_h, scale_w), order=0)   # nearest for masks


def detect_suffix(folder, patient_id):
    """
    BraTS 2020 datasets from different sources occasionally ship with either
    no suffix or a '.gz' suffix only. This helper returns the correct full
    filename for a modality so we are not hard-coded to one variant.
    """
    for ext in [".nii.gz", ".nii"]:
        candidate = os.path.join(folder, f"{patient_id}_flair{ext}")
        if os.path.isfile(candidate):
            return ext
    return ".nii.gz"   # fall back to the standard


# ─── GATHER PATIENT FOLDERS ───────────────────────────────────────────────────
all_folders = sorted([
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, f))
])

# Keep only folders that have a segmentation file (training split only)
valid_folders = []
for folder in all_folders:
    patient_id = os.path.basename(folder)
    ext = detect_suffix(folder, patient_id)
    seg_path = os.path.join(folder, f"{patient_id}_seg{ext}")
    if os.path.isfile(seg_path):
        valid_folders.append((folder, ext))

print(f"Found {len(valid_folders)} BraTS-2020 patients with segmentation masks")
print(f"Saving preprocessed slices to: {SAVE_PATH}")
print("This only needs to run ONCE — it takes ~10-20 minutes.")
print("=" * 60)

saved_count  = 0
skipped_count = 0

for folder, ext in tqdm(valid_folders, desc="Processing BraTS-2020 patients"):
    try:
        patient_id = os.path.basename(folder)

        def load(mod):
            return nib.load(
                os.path.join(folder, f"{patient_id}_{mod}{ext}")
            ).get_fdata()

        flair = load("flair")
        t1    = load("t1")
        t1ce  = load("t1ce")
        t2    = load("t2")
        seg   = load("seg")

        # Normalise each modality independently
        flair = normalize(flair)
        t1    = normalize(t1)
        t1ce  = normalize(t1ce)
        t2    = normalize(t2)

        # Resize spatial dims to IMG_SIZE × IMG_SIZE
        flair = resize_volume(flair, IMG_SIZE)
        t1    = resize_volume(t1,    IMG_SIZE)
        t1ce  = resize_volume(t1ce,  IMG_SIZE)
        t2    = resize_volume(t2,    IMG_SIZE)
        seg   = resize_volume(seg,   IMG_SIZE)

        # Iterate over axial slices (axis=2)
        for s in range(seg.shape[2]):
            mask_slice = seg[:, :, s]

            # Skip slices that contain no tumour annotation
            if np.sum(mask_slice) == 0:
                continue

            # Stack modalities → (4, 128, 128) float32
            img_slice = np.stack([
                flair[:, :, s],
                t1   [:, :, s],
                t1ce [:, :, s],
                t2   [:, :, s],
            ], axis=0).astype(np.float32)

            # Remap label 4 (enhancing tumour) → 3  (same convention as 2021)
            mask_slice = np.where(mask_slice == 4, 3, mask_slice).astype(np.int64)

            # Prefix with "brats20_" to avoid collisions with 2021 filenames
            prefix = f"brats20_{patient_id}_slice{s:03d}"
            np.save(os.path.join(SAVE_PATH, f"{prefix}_img.npy"),  img_slice)
            np.save(os.path.join(SAVE_PATH, f"{prefix}_mask.npy"), mask_slice)
            saved_count += 1

        del flair, t1, t1ce, t2, seg

    except Exception as e:
        print(f"\nSkipping {folder}: {e}")
        skipped_count += 1

print(f"\nDone!")
print(f"  Slices saved : {saved_count}")
print(f"  Patients skipped (errors): {skipped_count}")
print(f"  Output folder: {SAVE_PATH}")
print(
    "\nNext step: copy (or symlink) your existing BraTS-2021 preprocessed "
    "slices into the same folder, then point get_dataloaders() at it."
)

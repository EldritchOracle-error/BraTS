"""
inspect_brats2020.py

Run this script to print the exact folder structure and file names
inside your BraTS 2020 dataset directory so we can fix the preprocessing.
"""

import os

DATA_PATH = r"D:\\AI_in_Organ_Design\\datasets\\BraTS_2020\\extracted"

print(f"Scanning: {DATA_PATH}")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Path does not exist: {DATA_PATH}")
    print("Please check the DATA_PATH variable.")
    exit()

top_level = sorted(os.listdir(DATA_PATH))
print(f"Top-level entries ({len(top_level)} total):")
for entry in top_level[:10]:   # show first 10
    full = os.path.join(DATA_PATH, entry)
    kind = "DIR " if os.path.isdir(full) else "FILE"
    print(f"  [{kind}]  {entry}")

if len(top_level) > 10:
    print(f"  ... and {len(top_level) - 10} more")

print()

# ── Drill into the first subdirectory found ───────────────────────────────────
first_dirs = [
    os.path.join(DATA_PATH, e)
    for e in top_level
    if os.path.isdir(os.path.join(DATA_PATH, e))
]

if not first_dirs:
    print("No subdirectories found at the top level.")
    print("Your .nii.gz files might be directly inside DATA_PATH, not in patient subfolders.")
    print()
    print("Files directly in DATA_PATH:")
    for f in sorted(os.listdir(DATA_PATH))[:20]:
        print(f"  {f}")
else:
    first_dir = first_dirs[0]
    print(f"Contents of first patient folder: {first_dir}")
    print("-" * 60)
    contents = sorted(os.listdir(first_dir))
    for f in contents:
        full_f = os.path.join(first_dir, f)
        size_kb = os.path.getsize(full_f) // 1024 if os.path.isfile(full_f) else 0
        kind = "DIR " if os.path.isdir(full_f) else f"FILE ({size_kb} KB)"
        print(f"  [{kind}]  {f}")

    print()

    # Check if there's a second level of nesting
    subdirs_in_first = [
        os.path.join(first_dir, e)
        for e in contents
        if os.path.isdir(os.path.join(first_dir, e))
    ]
    if subdirs_in_first:
        print(f"  (nested subfolder found: {subdirs_in_first[0]})")
        print(f"  Contents of nested subfolder:")
        for f in sorted(os.listdir(subdirs_in_first[0]))[:10]:
            print(f"    {f}")

print()
print("=" * 60)
print("Copy the output above and share it to fix the preprocessing script.")

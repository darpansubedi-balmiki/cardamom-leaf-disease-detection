import os
from pathlib import Path

BASE = Path("dataset")
SPLITS = ["train", "val", "test"]
CLASSES = ["colletotrichum_blight", "healthy", "phyllosticta_leaf_spot"]

def get_files(split, cls):
    p = BASE / split / cls
    if not p.exists():
        return set()
    return set([f for f in os.listdir(p) if not f.startswith(".")])

for cls in CLASSES:
    files = {split: get_files(split, cls) for split in SPLITS}
    print("\nClass:", cls)
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            overlap = files[a] & files[b]
            print(f"{a} ∩ {b}: {len(overlap)}")
            if len(overlap) > 0:
                # Print a few examples
                print("  examples:", list(sorted(overlap))[:5])
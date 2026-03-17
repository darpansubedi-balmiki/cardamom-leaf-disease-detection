from pathlib import Path
from PIL import Image
import imagehash

DATASET_DIR = Path("dataset")
SPLITS = ["train", "val", "test"]

# Lower = stricter matching. 0 means identical. Try 2-5 first.
HAMMING_THRESHOLD = 3

def iter_images(split):
    root = DATASET_DIR / split
    for p in root.rglob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            yield p

def phash(path):
    # Convert to RGB to avoid mode issues
    with Image.open(path) as img:
        img = img.convert("RGB")
        return imagehash.phash(img)

def main():
    # Compute hashes per split
    hashes = {s: [] for s in SPLITS}
    print("Computing perceptual hashes... (this can take a bit)")

    for split in SPLITS:
        for p in iter_images(split):
            try:
                hashes[split].append((p, phash(p)))
            except Exception as e:
                print(f"Skip {p} ({e})")

        print(f"{split}: {len(hashes[split])} images hashed")

    # Compare splits
    def compare(a, b):
        print(f"\nChecking near-duplicates: {a} vs {b} (threshold={HAMMING_THRESHOLD})")
        matches = []
        for pa, ha in hashes[a]:
            for pb, hb in hashes[b]:
                d = ha - hb  # Hamming distance
                if d <= HAMMING_THRESHOLD:
                    matches.append((d, pa, pb))
        matches.sort(key=lambda x: x[0])
        print(f"Found {len(matches)} near-duplicate pairs.")
        for d, pa, pb in matches[:30]:  # print first 30
            print(f"  d={d}  {pa}  <->  {pb}")
        return matches

    compare("train", "val")
    compare("train", "test")
    compare("val", "test")

if __name__ == "__main__":
    main()
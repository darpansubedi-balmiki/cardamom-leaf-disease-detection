# Dataset Preparation Guide

This guide walks you through preparing a consistent training dataset where **all images have black backgrounds** (background removed), so the model learns leaf and disease features rather than background colours.

---

## Prerequisites

Make sure you are inside the `backend/` directory for all commands:

```bash
cd backend
```

---

## Step 1 — Install rembg

```bash
pip install rembg
```

> `rembg` uses the [U²-Net](https://github.com/xuebinqin/U-2-Net) model internally and downloads its weights automatically on first run (~170 MB).

---

## Step 2 — Remove backgrounds from each class folder

Run `remove_bg_batch.py` once per class. The script:
- Reads images from `--input`
- Removes the background with rembg
- Composites the result onto a solid black background
- Saves as JPEG (quality 95) to `--output`
- Skips files that already exist (resume-safe)

### Cardamom-specific classes

```bash
# Healthy cardamom leaves
python remove_bg_batch.py \
    --input dataset_raw/Healthy_1000 \
    --output dataset_processed/healthy \
    --limit 500

# Phyllosticta leaf spot
python remove_bg_batch.py \
    --input dataset_raw/Phyllosticta_LS_1000 \
    --output dataset_processed/phyllosticta_leaf_spot \
    --limit 500
```

> **Colletotrichum blight** images already have black backgrounds — skip this step for that class and go directly to Step 3.

### Other-class sub-folders

Run for each sub-folder in `dataset_raw/other/`. The `--output` for all of them is the same `dataset_processed/other` folder so they are merged into one class:

```bash
python remove_bg_batch.py --input dataset_raw/other/Apple       --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Berry       --output dataset_processed/other --limit 40
python remove_bg_batch.py --input dataset_raw/other/Fig         --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Guava       --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Orange      --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Palm        --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Persimmon   --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Tomato      --output dataset_processed/other --limit 40
```

Total `other` images: 60+40+60+60+60+60+60+40 = **440**

---

## Step 3 — Augment the blight class

The blight class has fewer source images than the other classes. Run `augment_blight.py` to bring it up to 500 images using black-background-safe transforms (no colour jitter, no blur):

```bash
# Edit SOURCE_DIR / OUTPUT_DIR at the top of augment_blight.py if needed, then:
python augment_blight.py
```

Default config inside the script:
| Variable | Default |
|---|---|
| `SOURCE_DIR` | `colletotrichum_blight` |
| `OUTPUT_DIR` | `colletotrichum_blight_augmented` |
| `TARGET_COUNT` | `500` |
| `SEED` | `42` |

---

## Step 4 — Split into train / val / test

```bash
python split_dataset.py
```

This creates `dataset/train/`, `dataset/val/`, and `dataset/test/` with a 70 / 15 / 15 split.

> Make sure the `SOURCE_FOLDERS` list in `split_dataset.py` points to the processed folders (e.g. `dataset_processed/healthy`, etc.) before running.

---

## Step 5 — Train the model

```bash
python train.py
```

The model will be saved to `models/cardamom_model.pt`.

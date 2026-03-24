# Dataset Preparation Guide

Follow these steps **IN ORDER** before running `python train.py`.

## Step 1 — Install rembg

```bash
pip install rembg
```

## Step 2 — Remove backgrounds from all raw images

### Disease classes

```bash
python remove_bg_batch.py --input dataset_raw/Blight1000 --output dataset_processed/colletotrichum_blight --limit 500
python remove_bg_batch.py --input dataset_raw/Healthy_1000 --output dataset_processed/healthy --limit 500
python remove_bg_batch.py --input dataset_raw/Phyllosticta_LS_1000 --output dataset_processed/phyllosticta_leaf_spot --limit 500
```

### Other class (subsample from each subfolder for variety)

```bash
python remove_bg_batch.py --input dataset_raw/other/Apple --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Berry --output dataset_processed/other --limit 40
python remove_bg_batch.py --input dataset_raw/other/Fig --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Guava --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Orange --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Palm --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Persimmon --output dataset_processed/other --limit 60
python remove_bg_batch.py --input dataset_raw/other/Tomato --output dataset_processed/other --limit 40
```

> **Note:** The `--output` folder is shared for all `other` subfolders. The `--limit` flag
> subsamples each source folder so the combined output is balanced. The script skips files
> that already exist (resume support), so you can safely re-run it.

## Step 3 — Augment Blight to balance classes

```bash
python augment_blight.py
```

This generates augmented blight images up to 500 total and saves them to
`dataset_processed/colletotrichum_blight_augmented/`.

> **Why no ColorJitter?** All images now have pure black backgrounds. Color jitter would
> tint the background and teach the model to look for non-black areas instead of leaf features.

## Step 4 — Split into train/val/test

```bash
python split_dataset.py
```

## Step 5 — Train

```bash
python train.py
```

---

## Background: Why background removal matters

Before this pipeline was introduced:

| Class | Background |
|---|---|
| `colletotrichum_blight` | Black (already removed) |
| `healthy` | Natural (green grass, soil, etc.) |
| `phyllosticta_leaf_spot` | Natural |
| `other` | Natural |

This inconsistency caused the model to learn **background colour** as a discriminative feature
rather than actual leaf or disease patterns. By normalising all classes to a black background,
the model is forced to focus on the leaf itself.

# Dataset

This project was fine-tuned on the **Doctor's Handwritten Prescription BD
dataset**. The dataset itself is not bundled with the repo -- it's a few
hundred MB of images and we don't want to push it through git.

## Where to get it

The dataset is on Kaggle:

- <https://www.kaggle.com/datasets/mehaksingal/doctors-handwritten-prescription-bd-dataset>

(There are a couple of mirrors of the same data. Any of them works -- the
file layout is the same.)

## Expected layout

After downloading and extracting, the folder should look like this and live
at the **repo root** (not inside `data/`), because that's where the
`/evaluate` route looks for it:

```
dataset/
└── Doctor's Handwritten Prescription BD dataset/
    ├── Training/
    │   ├── training_labels.csv          # IMAGE,MEDICINE_NAME,GENERIC_NAME
    │   └── training_words/              # one .png per word crop
    ├── Validation/
    │   ├── validation_labels.csv
    │   └── validation_words/
    └── Testing/
        ├── testing_labels.csv
        └── testing_words/
```

(The folder name has a unicode curly apostrophe in `Doctor's`. Keep it as-is
when extracting -- the code finds the path with `rglob`, so the exact name
doesn't matter, but it does need to live under `dataset/.../Testing/`.)

## What's actually in here

The two CSVs you'll touch most often:

| File | Columns |
|------|---------|
| `Training/training_labels.csv` | `IMAGE`, `MEDICINE_NAME`, `GENERIC_NAME` |
| `Testing/testing_labels.csv`  | `IMAGE`, `MEDICINE_NAME`, `GENERIC_NAME` |

Each `IMAGE` is a filename in the corresponding `*_words/` folder, and each
of those is a single tightly-cropped word -- a medicine name written out by
a doctor.

## sample_inputs/

A few representative test crops are checked in under `sample_inputs/` so you
can try the `/predict` upload form without downloading the whole dataset.

These are from `Testing/testing_words/`. They're literally just five
hand-picked PNGs -- enough to see the OCR work end-to-end.

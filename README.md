# AI Prescription Reader

A small Flask app that reads handwritten doctor's prescriptions, figures out
which medicines are written on the page, and warns you about potential
drug-drug interactions.

Behind the scenes:

1. **Word segmentation** -- a classical OpenCV pipeline (adaptive threshold +
   horizontal morphology + connected components) splits the page into
   word-level crops in reading order.
2. **OCR** -- each crop is fed through Microsoft's TrOCR. We actually run
   *two* models side-by-side:
     - a fine-tune trained on the "Doctor's Handwritten Prescription BD"
       dataset, which is great on familiar drug names, and
     - the stock `microsoft/trocr-base-handwritten`, which often does better
       on words it's never seen before.
   For each word we keep whichever of the two gets a stronger RxNorm hit.
3. **Drug matching** -- we ask the public RxNav `approximateTerm` API which
   medicine the OCR text is closest to, and how confident it is.
4. **Interaction check** -- if two or more words match real drugs, we ask
   RxNav's `interaction/list` endpoint whether they're known to interact.

Everything is exposed through a small Flask UI: upload a single word crop,
upload a whole prescription page, or run a benchmark sweep against the
held-out test split.

---

## Repository layout

```
AI-Prescription-Reader/
├── README.md
├── requirements.txt
├── app.py                         # Flask app -- routes + glue only
│
├── models/
│   └── trocr_model/               # fine-tuned TrOCR weights (see below)
│
├── data/
│   ├── sample_inputs/             # a handful of test crops you can try
│   └── README.md                  # where the full dataset comes from
│
├── src/
│   ├── preprocessing.py           # OpenCV page -> word-crop segmentation
│   ├── ocr_pipeline.py            # TrOCR loading + batched inference
│   ├── drug_matching.py           # RxNorm approximate-term + CER helpers
│   └── interaction_checker.py     # RxNorm drug-drug interactions
│
├── results/
│   ├── metrics.png                # accuracy / CER chart
│   └── sample_outputs/            # example predictions vs. ground truth
│
├── docs/
│   ├── architecture.png           # pipeline diagram
│   └── Final_Report.docx          # write-up (optional)
│
├── demo/
│   └── demo_images/               # a full prescription page for the demo
│
├── static/   templates/           # Flask static files + Jinja templates
└── Final_Code.ipynb               # the training notebook (TrOCR fine-tune)
```

The notebook (`Final_Code.ipynb`) is what produced the weights in
`models/trocr_model/`. It isn't needed at runtime -- it's there for reference
and so the fine-tune is reproducible.

---

## Setup

The app was developed against **Python 3.12** on Windows, but it should work
on any 3.10+ install with Python and pip on the path.

### 1. Create a virtual environment

```bash
# Windows (PowerShell or Git Bash)
python -m venv venv
venv/Scripts/python.exe -m pip install --upgrade pip

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch (CPU build)

PyTorch isn't pinned in `requirements.txt` because the right wheel depends on
your machine. For a plain CPU install:

```bash
# Windows
venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch

# Linux / macOS
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

If you have a CUDA-capable GPU and want to use it, grab the matching
`+cu12x` wheel from <https://pytorch.org/get-started/locally/> instead.
The code auto-detects CUDA -- nothing else needs to change.

### 3. Install the rest

```bash
venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
pip install -r requirements.txt                              # Linux / macOS
```

That pulls in Flask, Pillow, transformers, OpenCV, pandas, requests.

---

## Running the app

One command:

```bash
venv/Scripts/python.exe app.py     # Windows
python app.py                      # Linux / macOS
```

Then open <http://127.0.0.1:5000>.

The first request takes a couple of seconds longer than the rest -- both
TrOCR models are loaded into memory at startup, so subsequent inferences
are quick.

### What the pages do

| Page | What it does |
|------|--------------|
| `/` | Upload form. Choose a single page, or one-or-more word crops. |
| `/predict` | OCRs each crop, looks each one up in RxNorm, runs an interaction check on whatever it matched. |
| `/predict-page` | Segments a whole prescription, OCRs every word, picks the ones that look like real medicines, runs interactions on them. |
| `/evaluate` | Runs the fine-tune on a seeded random sample of the test split. Reports exact-match % and average CER. |

There's a `demo/demo_images/test_full_prescription.jpg` you can drop into
the page-upload form to try the full pipeline end-to-end.

---

## Pre-trained model

The fine-tuned weights live in `models/trocr_model/`. They're ~1.3 GB --
too big to ship in a normal git repo, so:

- If you cloned this and the folder is present, you're good.
- If it's missing or empty, the app falls back to `microsoft/trocr-base-handwritten`
  from the Hugging Face Hub on first run. The UI will show
  *"base pretrained (no fine-tuned checkpoint found)"* so you know.
- To regenerate the fine-tune from scratch, run `Final_Code.ipynb` end to
  end. It will write the weights back to `models/trocr_model/`.

If you want to publish the weights separately, dropping the folder onto a
release / Hugging Face / Drive link and pointing people at it is fine -- the
loader in `src/ocr_pipeline.py` only cares that the directory exists and is
non-empty.

---

## Dataset

The fine-tune and the `/evaluate` page both use the **Doctor's Handwritten
Prescription BD dataset**. It's not bundled here -- see
[`data/README.md`](data/README.md) for the source link and how to lay it out
on disk.

A handful of test crops are checked in under `data/sample_inputs/` so you
can try the upload UI without downloading the whole dataset.

---

## Results

A few of the visualisations from the training notebook:

- `results/metrics.png` -- predictions vs. ground truth
- `results/training_loss_curve.png` -- the fine-tune loss curve
- `results/inference_times.png` -- latency per crop, fine-tune vs. base
- `results/sample_outputs/` -- side-by-side examples

`/evaluate` runs the same kind of comparison live, with a seed so consecutive
runs are directly comparable.

---

## File-by-file notes

- **`app.py`** -- Flask app and routes only. No model code, no parsing logic.
- **`src/preprocessing.py`** -- segmentation. The `SegmentParams` dataclass
  is the place to twiddle if your scans look different from the training
  data.
- **`src/ocr_pipeline.py`** -- the `OCREngine` class wraps both TrOCR models.
  It's instantiated once when `app.py` is imported.
- **`src/drug_matching.py`** -- RxNorm `approximateTerm` lookup, plus
  Levenshtein / CER for the eval metric.
- **`src/interaction_checker.py`** -- RxNorm `interaction/list` lookup.
  Flattens the (deeply nested) response into a list of pairs.

---

## Troubleshooting

- **"Loading fine-tuned TrOCR..." hangs at startup.** First run downloads
  the base model from Hugging Face -- needs internet for that. After the
  download is cached (in `~/.cache/huggingface/`) it's instant.
- **Segmentation finds 0 boxes.** Your scan is probably very faint or very
  noisy. Bump `adaptive_C` down or `adaptive_block` up in
  `src/preprocessing.py`.
- **RxNorm matches all empty.** RxNav is occasionally flaky. Refresh, or
  check <https://rxnav.nlm.nih.gov/> directly. Network errors are swallowed
  on purpose -- we'd rather show "no match" than crash the page.
- **`MAX_CONTENT_LENGTH` exceeded.** Uploads are capped at 32 MB. Resize
  the image, or bump the limit at the top of `app.py`.

---

## Acknowledgements

- Base OCR: [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten)
- Drug data: [RxNorm / RxNav](https://rxnav.nlm.nih.gov/) (public, no key needed)
- Dataset: Doctor's Handwritten Prescription BD (Kaggle -- see `data/README.md`)

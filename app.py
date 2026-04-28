"""
Flask entry point for the AI Prescription Reader.

This file is intentionally thin -- it's just the web layer. The real work
lives in src/:
    src/preprocessing.py      -> page -> word crops
    src/ocr_pipeline.py       -> TrOCR (fine-tuned + base)
    src/drug_matching.py      -> RxNorm approximate-term lookup + CER helpers
    src/interaction_checker.py -> RxNorm drug-drug interaction lookup

Routes:
    /              upload page
    /predict       POST one-or-more single-word crops
    /predict-page  POST a full prescription page (segmented + OCR'd)
    /evaluate      benchmark the fine-tune on the test split

Run it with:  python app.py
Open:         http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from src.ocr_pipeline import OCREngine, DEVICE
from src.preprocessing import segment_word_crops
from src.drug_matching import (
    rxnorm_approximate,
    candidate_score,
    char_error_rate,
)
from src.interaction_checker import rxnorm_interactions


# ---------- Paths and constants ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# What the upload form will accept. Keep this in sync with the input element
# in templates/index.html if you change it.
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------- Models -------------------------------------------------------
# Loaded once at import time. This takes a few seconds on a cold start but
# means every request is fast. There are two TrOCR models loaded -- see
# src/ocr_pipeline.py for the rationale.

ENGINE = OCREngine()
MODEL_SOURCE = ENGINE.ft_source
MODEL_KIND = ENGINE.ft_kind


# ---------- Helpers ------------------------------------------------------

def _save_upload(file_storage, prefix: str = "") -> tuple[Path, str]:
    """Save an uploaded file under static/uploads/ with a unique-ish name.

    Returns (path-on-disk, filename-only). The 12-char uuid prefix avoids
    collisions when two users upload "scan.jpg" at the same time.
    """
    safe = secure_filename(file_storage.filename) or "upload"
    unique = f"{prefix}{uuid.uuid4().hex[:12]}_{safe}"
    saved = UPLOAD_DIR / unique
    file_storage.save(saved)
    return saved, unique


def _find_test_paths():
    """Locate the test CSV + image dir under dataset/.

    The dataset folder name has a unicode curly apostrophe in it, so we use
    rglob rather than hardcoding the path. Returns (None, None) if the
    dataset isn't present (e.g. user hasn't downloaded it yet).
    """
    dataset_root = BASE_DIR / "dataset"
    if not dataset_root.exists():
        return None, None
    test_dir = next(dataset_root.rglob("Testing"), None)
    if test_dir is None:
        return None, None
    csv = next(test_dir.rglob("*_labels.csv"), None)
    img_dir = next(test_dir.rglob("testing_words"), None)
    if csv is None or img_dir is None:
        return None, None
    return csv, img_dir


def _pick_better(ft_top, base_top):
    """Of the two RxNorm top-candidates, return the one with the higher score.

    Ties go to the fine-tune. Returns (best_side_label, best_top_dict,
    ft_score, base_score).
    """
    ft_score = candidate_score(ft_top)
    base_score = candidate_score(base_top)
    if ft_score >= base_score:
        return "fine-tune", ft_top, ft_score, base_score
    return "base", base_top, ft_score, base_score


# ---------- App factory --------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)

    # FLASK_SECRET should be set in production. The dev default is fine for
    # localhost demos but obviously not for anything else.
    app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

    # Cap uploads at 32 MB. Anything bigger is almost certainly a mistake.
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            model_source=MODEL_SOURCE,
            model_kind=MODEL_KIND,
            device=str(DEVICE),
        )

    # ------------------------------------------------------------------
    # /predict -- one or more single-word crops
    # ------------------------------------------------------------------

    @app.route("/predict", methods=["POST"])
    def predict_route():
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not files:
            flash("No files uploaded.")
            return redirect(url_for("index"))

        results = []
        for f in files:
            ext = Path(f.filename).suffix.lower()
            if ext not in ALLOWED_EXT:
                results.append({
                    "filename": None,
                    "original": f.filename,
                    "error": f"Unsupported file type: {ext}",
                })
                continue

            saved, unique_name = _save_upload(f)

            try:
                pil = Image.open(saved).convert("RGB")
                # Time both inferences -- the UI shows them so people can
                # eyeball the speed difference.
                t0 = time.perf_counter()
                pred_ft = ENGINE.predict_one_ft(pil)
                ft_ms = int((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                pred_base = ENGINE.predict_one_base(pil)
                base_ms = int((time.perf_counter() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                # We don't want one bad image to take down the whole batch.
                results.append({
                    "filename": unique_name,
                    "original": f.filename,
                    "error": str(e),
                })
                continue

            ft_cands = rxnorm_approximate(pred_ft)
            base_cands = rxnorm_approximate(pred_base)
            ft_top = ft_cands[0] if ft_cands else None
            base_top = base_cands[0] if base_cands else None

            best_side, best_top, ft_score, base_score = _pick_better(ft_top, base_top)

            results.append({
                "filename": unique_name,
                "original": f.filename,
                "ft_prediction": pred_ft,
                "base_prediction": pred_base,
                "ft_match": ft_top,
                "base_match": base_top,
                "ft_score": ft_score,
                "base_score": base_score,
                "best_side": best_side,
                "best_top": best_top,
                "ft_ms": ft_ms,
                "base_ms": base_ms,
                "ft_candidates": ft_cands,
                "base_candidates": base_cands,
            })

        # Pull out the rxcuis we actually picked, then check interactions.
        rxcuis = [
            r["best_top"]["rxcui"]
            for r in results
            if r.get("best_top") and r["best_top"].get("rxcui")
        ]
        interactions = rxnorm_interactions(rxcuis)

        return render_template(
            "result.html",
            results=results,
            interactions=interactions,
            checked_interactions=len(rxcuis) >= 2,
        )

    # ------------------------------------------------------------------
    # /predict-page -- a full prescription page
    # ------------------------------------------------------------------

    @app.route("/predict-page", methods=["POST"])
    def predict_page_route():
        f = request.files.get("image")
        if not f or not f.filename:
            flash("No image uploaded.")
            return redirect(url_for("index"))
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            flash(f"Unsupported file type: {ext}")
            return redirect(url_for("index"))

        saved, unique = _save_upload(f)

        # Optional confidence threshold from the form. Anything we can't
        # parse is treated as 0 (i.e. accept everything).
        try:
            min_score = float(request.form.get("min_score", 0.0))
        except ValueError:
            min_score = 0.0

        try:
            page = Image.open(saved).convert("RGB")
        except Exception as e:  # noqa: BLE001
            flash(f"Could not read image: {e}")
            return redirect(url_for("index"))

        # Step 1 -- segment into word crops.
        t0 = time.perf_counter()
        seg = segment_word_crops(page)
        seg_ms = int((time.perf_counter() - t0) * 1000)

        # Save the annotated preview so the template can show it.
        annotated_name = f"annotated_{unique}.png"
        seg.annotated.save(UPLOAD_DIR / annotated_name)

        crops = [b.crop for b in seg.boxes]

        # Step 2 -- batch OCR with both models.
        t0 = time.perf_counter()
        preds_ft = ENGINE.predict_ft(crops, batch_size=8) if crops else []
        ocr_ft_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        preds_base = ENGINE.predict_base(crops, batch_size=8) if crops else []
        ocr_base_ms = int((time.perf_counter() - t0) * 1000)

        # Step 3 -- per-word RxNorm match + interaction check on the keepers.
        words = []
        medicines = []
        for b, p_ft, p_base in zip(seg.boxes, preds_ft, preds_base):
            # Save each crop too so the result page can display it.
            crop_name = f"crop_{unique}_{b.index:03d}.png"
            b.crop.save(UPLOAD_DIR / crop_name)

            cands_ft = rxnorm_approximate(p_ft)
            cands_base = rxnorm_approximate(p_base)
            top_ft = cands_ft[0] if cands_ft else None
            top_base = cands_base[0] if cands_base else None

            best_side, best_top, score_ft, score_base = _pick_better(top_ft, top_base)
            best_score = max(score_ft, score_base)
            best_pred = p_ft if best_side == "fine-tune" else p_base

            # If the score clears the threshold we treat this word as a real
            # medicine name (otherwise it's probably "Patient" or "Mr." or
            # the date line).
            is_medicine = best_top is not None and best_score >= min_score

            item = {
                "index": b.index + 1,
                "bbox": b.bbox,
                "crop_name": crop_name,
                "ft_prediction": p_ft,
                "ft_match": top_ft,
                "ft_score": score_ft,
                "base_prediction": p_base,
                "base_match": top_base,
                "base_score": score_base,
                "best_side": best_side,
                "best_pred": best_pred,
                "best_top": best_top,
                "best_score": best_score,
                "is_medicine": is_medicine,
            }
            words.append(item)
            if is_medicine:
                medicines.append(item)

        rxcuis = [
            m["best_top"]["rxcui"]
            for m in medicines
            if m.get("best_top") and m["best_top"].get("rxcui")
        ]
        interactions = rxnorm_interactions(rxcuis)

        return render_template(
            "result_page.html",
            original_name=unique,
            annotated_name=annotated_name,
            words=words,
            medicines=medicines,
            interactions=interactions,
            seg_ms=seg_ms,
            ocr_ms=ocr_ft_ms + ocr_base_ms,
            n_boxes=len(seg.boxes),
            min_score=min_score,
        )

    # ------------------------------------------------------------------
    # /evaluate -- benchmark on the held-out test split
    # ------------------------------------------------------------------

    @app.route("/evaluate", methods=["GET", "POST"])
    def evaluate_route():
        csv_path, img_dir = _find_test_paths()
        has_dataset = csv_path is not None

        if request.method == "POST" and has_dataset:
            try:
                sample_size = max(1, int(request.form.get("sample_size", 20)))
            except ValueError:
                sample_size = 20

            df = pd.read_csv(csv_path)
            # random_state=42 is intentional -- we want consecutive runs to
            # be comparable, not a fresh shuffle every time.
            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            rows = []
            exact_hits = 0
            total_cer = 0.0
            t0 = time.perf_counter()
            for _, r in df.iterrows():
                img_path = img_dir / str(r["IMAGE"])
                if not img_path.exists():
                    # Skip silently -- the CSV occasionally references files
                    # that aren't on disk.
                    continue
                gt = str(r["MEDICINE_NAME"]).strip()
                try:
                    pred = ENGINE.predict_one_path(img_path)
                except Exception:  # noqa: BLE001
                    continue

                # Lower-case both for the metric -- the model is case-aware,
                # but for "did we get the medicine right?" case doesn't matter.
                is_exact = pred.lower() == gt.lower()
                cer = char_error_rate(pred.lower(), gt.lower())
                if is_exact:
                    exact_hits += 1
                total_cer += cer
                rows.append({
                    "image": r["IMAGE"],
                    "ground_truth": gt,
                    "prediction": pred,
                    "exact": is_exact,
                    "cer": round(cer, 3),
                })
            elapsed = time.perf_counter() - t0

            n = len(rows)
            summary = {
                "n": n,
                "exact_match_pct": round(100 * exact_hits / n, 2) if n else 0.0,
                "avg_cer": round(total_cer / n, 4) if n else 0.0,
                "elapsed_sec": round(elapsed, 1),
            }
            return render_template(
                "evaluate.html",
                has_dataset=True,
                ran=True,
                summary=summary,
                rows=rows,
                total_available=len(pd.read_csv(csv_path)),
            )

        # GET (or POST without the dataset present) -- just render the form.
        total_available = len(pd.read_csv(csv_path)) if has_dataset else 0
        return render_template(
            "evaluate.html",
            has_dataset=has_dataset,
            ran=False,
            total_available=total_available,
        )

    return app


# Module-level app object, picked up by `flask run` and by `python app.py`.
app = create_app()


if __name__ == "__main__":
    # debug=False because we have two TrOCR models in memory -- the auto
    # reloader would happily double them.
    app.run(debug=False, host="127.0.0.1", port=5000)

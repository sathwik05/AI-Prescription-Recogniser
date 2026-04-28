"""
TrOCR loading + inference helpers.

We deliberately load TWO models at startup:
  - the fine-tuned checkpoint (best on medicine names it has seen)
  - the plain microsoft/trocr-base-handwritten (better on out-of-vocab words)

Each prediction route runs both, and the rest of the app picks whichever one
returns a stronger RxNorm match. A bit wasteful CPU-wise, but on a single-user
demo it's fine and the extra signal is useful.

If models/trocr_model/ doesn't exist (e.g. you cloned without the weights),
the fine-tune slot quietly falls back to the base model. The UI will still
render -- the comparison just becomes uninformative.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Where the fine-tuned weights live. Changed from checkpoints/best_model
# (old layout) to models/trocr_model.
DEFAULT_FT_DIR = Path(__file__).resolve().parents[1] / "models" / "trocr_model"
FALLBACK_MODEL = "microsoft/trocr-base-handwritten"

# 64 tokens is plenty -- medicine names are usually short, and longer
# generations just slow things down.
MAX_TARGET_LENGTH = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_one(src: str) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load a single (processor, model) pair and move it to DEVICE."""
    processor = TrOCRProcessor.from_pretrained(src)
    model = VisionEncoderDecoderModel.from_pretrained(src)
    model.to(DEVICE).eval()
    return processor, model


def load_finetuned(ft_dir: Path | None = None) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel, str, str]:
    """Load the fine-tune, falling back to the base model if the dir is empty.

    Returns (processor, model, source_path_or_id, kind_label).
    The kind label is just a human string so we can show it in the UI.
    """
    ft_dir = ft_dir or DEFAULT_FT_DIR
    if ft_dir.exists() and any(ft_dir.iterdir()):
        src = str(ft_dir)
        kind = "fine-tuned checkpoint"
    else:
        src = FALLBACK_MODEL
        kind = "base pretrained (no fine-tuned checkpoint found)"
    proc, mdl = _load_one(src)
    return proc, mdl, src, kind


def load_base() -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Always-on base model. Used for the side-by-side comparison."""
    return _load_one(FALLBACK_MODEL)


class OCREngine:
    """Holds the two models + a couple of small batch-prediction helpers.

    Centralising the models in a class means callers don't have to juggle
    separate `(processor_ft, model_ft, processor_base, model_base)` tuples.
    """

    def __init__(self, ft_dir: Path | None = None):
        print("Loading fine-tuned TrOCR...", flush=True)
        self.proc_ft, self.model_ft, self.ft_source, self.ft_kind = load_finetuned(ft_dir)
        print(f"  -> {self.ft_source} ({self.ft_kind}) on {DEVICE}", flush=True)

        print("Loading base TrOCR (microsoft/trocr-base-handwritten)...", flush=True)
        self.proc_base, self.model_base = load_base()
        print(f"  -> {FALLBACK_MODEL} on {DEVICE}", flush=True)

    # --- internal generation helper ----------------------------------------

    def _generate(self, processor, model, images: List[Image.Image], batch_size: int) -> List[str]:
        """Run TrOCR on a list of PIL images, in batches, return the strings."""
        out: List[str] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            pixel_values = processor(
                images=[im.convert("RGB") for im in chunk],
                return_tensors="pt",
            ).pixel_values.to(DEVICE)
            # No grad -- this is pure inference. Saves memory, runs faster.
            with torch.no_grad():
                ids = model.generate(pixel_values, max_length=MAX_TARGET_LENGTH)
            out.extend(s.strip() for s in processor.batch_decode(ids, skip_special_tokens=True))
        return out

    # --- public API used by app.py -----------------------------------------

    def predict_ft(self, images: List[Image.Image], batch_size: int = 8) -> List[str]:
        """Fine-tuned recognizer."""
        return self._generate(self.proc_ft, self.model_ft, images, batch_size)

    def predict_base(self, images: List[Image.Image], batch_size: int = 8) -> List[str]:
        """Base recognizer -- often better on words the fine-tune hasn't seen."""
        return self._generate(self.proc_base, self.model_base, images, batch_size)

    def predict_one_ft(self, image: Image.Image) -> str:
        return self.predict_ft([image], batch_size=1)[0]

    def predict_one_base(self, image: Image.Image) -> str:
        return self.predict_base([image], batch_size=1)[0]

    def predict_one_path(self, path: Path) -> str:
        """Fine-tune prediction from a file path -- handy for the eval loop."""
        return self.predict_one_ft(Image.open(path).convert("RGB"))

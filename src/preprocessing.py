"""
Word segmentation for full prescription pages.

The TrOCR model was fine-tuned on single-word crops (one medicine name per
image), so a full prescription page won't work directly -- we need to chop it
into word-level pieces first. That's what this file does, using plain old
OpenCV: adaptive threshold, a bit of morphology to glue characters into words,
then connected components to pull out the boxes.

Knobs are exposed in SegmentParams because handwritten scans vary so much --
ink darkness, paper texture, scan resolution, you name it. If a particular
prescription comes out wrong, tweaking these here is the right place; don't
re-implement the threshold inside app.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SegmentParams:
    # Block size for the adaptive threshold. Must be odd; we'll fix it if
    # someone passes an even number.
    adaptive_block: int = 31
    # Higher C -> stricter threshold (less noise, but you can lose faint ink).
    adaptive_C: int = 15

    # Tiny opening to kill speckle noise before we dilate.
    denoise_kernel: Tuple[int, int] = (2, 2)

    # Wide-but-short kernel: glues neighbouring characters into one word blob
    # without merging across rows.
    word_kernel: Tuple[int, int] = (25, 5)
    word_iters: int = 2

    # Reject obviously-not-a-word boxes.
    min_w: int = 25
    min_h: int = 14
    min_area: int = 400
    max_w_frac: float = 0.9    # box wider than 90% of the page is junk
    max_h_frac: float = 0.25   # box taller than a quarter of the page is junk

    # Padding around each crop so the OCR sees a little breathing room.
    pad: int = 6

    # Two boxes whose top-y differs by less than this are considered same row.
    row_tolerance: int = 25


@dataclass
class WordBox:
    crop: Image.Image
    bbox: Tuple[int, int, int, int]   # (x0, y0, x1, y1)
    index: int                        # reading order, 0-based


@dataclass
class SegmentResult:
    boxes: List[WordBox]
    annotated: Image.Image            # original page with numbered boxes drawn
    params: SegmentParams = field(default_factory=SegmentParams)


def _sort_reading_order(rects, row_tolerance: int):
    """Top-to-bottom, then left-to-right.

    We do this in two passes -- bucket boxes into rows by their top-y, then
    sort each row by x. A flat sort by (y, x) doesn't work because two words
    on the same line often have y values that differ by a few pixels.
    """
    rects = sorted(rects, key=lambda r: r[1])
    rows = []
    for r in rects:
        placed = False
        for row in rows:
            # Compare against the first box in the row -- it's "the row's y".
            if abs(row[0][1] - r[1]) <= row_tolerance:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])

    ordered = []
    for row in rows:
        row.sort(key=lambda r: r[0])
        ordered.extend(row)
    return ordered


def segment_word_crops(
    pil_image: Image.Image,
    params: SegmentParams | None = None,
) -> SegmentResult:
    """Take a page, return word-level crops in reading order + a preview image."""
    p = params or SegmentParams()

    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # cv2 wants an odd block size -- bump it up if needed.
    block = p.adaptive_block if p.adaptive_block % 2 == 1 else p.adaptive_block + 1

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=p.adaptive_C,
    )

    # Quick open to clean up tiny specks.
    kn = cv2.getStructuringElement(cv2.MORPH_RECT, p.denoise_kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kn)

    # Horizontal dilation: turn separate characters into a single blob per word.
    kw = cv2.getStructuringElement(cv2.MORPH_RECT, p.word_kernel)
    merged = cv2.dilate(thr, kw, iterations=p.word_iters)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # Filter junk -- too small, too thin, or stretching across the page.
        if cw < p.min_w or ch < p.min_h:
            continue
        if cw * ch < p.min_area:
            continue
        if cw > w * p.max_w_frac:
            continue
        if ch > h * p.max_h_frac:
            continue
        rects.append((x, y, cw, ch))

    rects = _sort_reading_order(rects, p.row_tolerance)

    boxes: List[WordBox] = []
    for idx, (x, y, cw, ch) in enumerate(rects):
        # Pad the crop a bit -- TrOCR is happier with some margin.
        x0 = max(0, x - p.pad)
        y0 = max(0, y - p.pad)
        x1 = min(w, x + cw + p.pad)
        y1 = min(h, y + ch + p.pad)
        crop = Image.fromarray(rgb[y0:y1, x0:x1])
        boxes.append(WordBox(crop=crop, bbox=(x0, y0, x1, y1), index=idx))

    # Draw a preview image so the user can sanity-check the segmentation.
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        # On Linux/macOS arial isn't always there -- fall back to whatever PIL has.
        font = ImageFont.load_default()

    for b in boxes:
        x0, y0, x1, y1 = b.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(37, 99, 235), width=2)
        # Number tag in the corner -- 1-indexed for humans.
        label = str(b.index + 1)
        tw = max(12, 8 * len(label) + 8)
        draw.rectangle([x0, max(0, y0 - 20), x0 + tw, y0], fill=(37, 99, 235))
        draw.text((x0 + 4, max(0, y0 - 20)), label, fill=(255, 255, 255), font=font)

    return SegmentResult(boxes=boxes, annotated=annotated, params=p)

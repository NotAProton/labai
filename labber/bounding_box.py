"""
bounding_box.py — Multi-stage annotation pipeline for labber.

Pipeline
--------
Stage 1 : Amazon Nova LLM  — resizes to 360 px short side, uses Nova's
          [0, 1000) grounding coordinate scale, prefill technique for
          structured JSON output.

Stage 2 : PaddleOCR        — runs on a padded crop of the full-resolution image.
          Uses deep-learning OCR to snap boundaries to exact text pixels,
          handling URLs, timestamps, and special characters natively.

Stage 3 : OpenCV refinement — snaps to UI grid lines and AXIOM selection rows.

Stage 4 : PIL drawing      — renders the final boxes.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, ImageDraw

# ── Optional deps ──────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    # Initialize singleton to avoid reloading the model on every call
    # show_log=False prevents Paddle from spamming the console
    _PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    _PADDLE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PADDLE_AVAILABLE = False
    _PADDLE_OCR = None

# ── Constants ──────────────────────────────────────────────────────────────
ANNOTATION_MODEL_ID: str = os.environ.get("ANNOTATION_MODEL_ID", "apac.amazon.nova-pro-v1:0")
ANNOTATION_REGION: str = os.environ.get("ANNOTATION_REGION", "ap-south-1")

REQUEST_DELAY: float = 0.5
ANNOTATION_COLOR: str = "#FF3333"
ANNOTATION_WIDTH: int = 3

BOX_AREA_MIN_FRAC: float = 0.001
BOX_AREA_MAX_FRAC: float = 0.50
NOVA_SHORT_SIDE: int = 360

_UNION_THRESHOLD: float = 0.50   
_ACCEPT_THRESHOLD: float = 0.60  

_STOPWORDS: frozenset[str] = frozenset({
    "the", "in", "of", "on", "at", "to", "a", "an", "for",
    "with", "and", "or", "is", "are", "was", "this", "that", "value",
    "pane", "field", "label", "column", "row", "evidence", "details",
    "next", "selected", "showing", "such", "as", "represent"
})

_ann_client: object | None = None
_ann_client_region: str | None = None

# ── Bedrock & JSON Helpers ─────────────────────────────────────────────────

def _get_ann_client(region: str | None = None):
    global _ann_client, _ann_client_region
    target = region or ANNOTATION_REGION
    if _ann_client is None or _ann_client_region != target:
        _ann_client = boto3.client("bedrock-runtime", region_name=target)
        _ann_client_region = target
    return _ann_client

def _safe_json_load(json_string: str) -> list[dict]:
    try:
        json_string = re.sub(r"\s", "", json_string)
        json_string = re.sub(r"\(", "[", json_string)
        json_string = re.sub(r"\)", "]", json_string)
        bbox_set: dict[str, tuple[int, int]] = {}
        for b in re.finditer(r"\[\d+,\d+,\d+,\d+\]", json_string):
            key = b.group(0)
            if key in bbox_set:
                json_string = json_string[: bbox_set[key][1]] + "}]"
                break
            bbox_set[key] = (b.start(), b.end())
        else:
            if bbox_set:
                last = max(bbox_set.values(), key=lambda x: x[1])
                json_string = json_string[: last[1]] + "}]"
        json_string = re.sub(r"\]\},\]$", "]}]", json_string)
        json_string = re.sub(r"\]\],\[\"", "]},{\"", json_string)
        json_string = re.sub(r"\]\],\[\{\"", "]},{\"", json_string)
        return json.loads(json_string)
    except Exception as exc:
        print(f"    [bbox] JSON parse warning: {exc}", file=sys.stderr)
        return []

def _pil_to_webp_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="webp", quality=quality)
    return buf.getvalue()

def _passes_sanity_gate(box: list[int], img_w: int, img_h: int) -> bool:
    left, top, right, bottom = box
    box_area = max(0, right - left) * max(0, bottom - top)
    img_area = img_w * img_h
    if img_area == 0 or box_area == 0: return False
    return BOX_AREA_MIN_FRAC <= (box_area / img_area) <= BOX_AREA_MAX_FRAC

def _nearest(values: list[int], target: int, prefer: str = "nearest") -> int:
    if not values: return target
    if prefer == "floor":
        cands = [v for v in values if v <= target]
        return max(cands) if cands else min(values, key=lambda v: abs(v - target))
    if prefer == "ceil":
        cands = [v for v in values if v >= target]
        return min(cands) if cands else min(values, key=lambda v: abs(v - target))
    return min(values, key=lambda v: abs(v - target))

# ── Stage 1: Nova LLM ──────────────────────────────────────────────────────

def _stage1_nova_boxes(client, cropped_img, annotation_recommendations, model_id):
    w_orig, h_orig = cropped_img.size
    short_side = min(w_orig, h_orig)
    if short_side > NOVA_SHORT_SIDE:
        ratio = NOVA_SHORT_SIDE / short_side
        small = cropped_img.resize((round(w_orig * ratio), round(h_orig * ratio)), Image.Resampling.LANCZOS)
    else:
        small = cropped_img

    img_bytes = _pil_to_webp_bytes(small)
    category_str = ", ".join(f'"{rec}"' for rec in annotation_recommendations)
    first_label = annotation_recommendations[0]

    prompt = (
        f"Detect bounding box of objects in the image, only detect "
        f"{category_str} category objects with high confidence, "
        f"output in a list of bounding box format.\nOutput example:\n"
        f'[\n    {{"{first_label}": [x1, y1, x2, y2]}},\n    ...\n]'
    )
    prefill = '[{\n    "'
    messages = [
        {"role": "user", "content": [{"image": {"format": "webp", "source": {"bytes": img_bytes}}}, {"text": prompt}]},
        {"role": "assistant", "content": [{"text": prefill}]},
    ]

    last_exc = None
    for attempt in range(3):
        if attempt > 0: time.sleep(1.0)
        try:
            resp = client.converse(modelId=model_id, messages=messages, inferenceConfig={"temperature": 0.0, "maxTokens": 1024}, serviceTier={'type': 'flex'})
            time.sleep(REQUEST_DELAY)
            raw = prefill + resp["output"]["message"]["content"][0]["text"]
            break
        except (ClientError, BotoCoreError) as exc:
            print(f"    [bbox/nova] call failed (attempt {attempt + 1}/3): {exc}", file=sys.stderr)
            last_exc = exc
    else:
        return []

    parsed = _safe_json_load(raw)
    results = []
    for item in parsed:
        if not isinstance(item, dict): continue
        label = next(iter(item))
        coords = item[label]
        if not (isinstance(coords, list) and len(coords) == 4): continue
        x1, y1, x2, y2 = (int(c) for c in coords)
        
        px1 = max(0, min(round(x1 / 1000 * w_orig), w_orig - 1))
        py1 = max(0, min(round(y1 / 1000 * h_orig), h_orig - 1))
        px2 = max(px1 + 1, min(round(x2 / 1000 * w_orig), w_orig))
        py2 = max(py1 + 1, min(round(y2 / 1000 * h_orig), h_orig))
        results.append((label, [px1, py1, px2, py2]))

    return results

# ── Stage 2: PaddleOCR Snapping ────────────────────────────────────────────

def _stage2_paddle_snap(img: Image.Image, box: list[int], recommendation: str) -> list[int]:
    """
    Snap the annotation rectangle using PaddleOCR.
    Handles asymmetric padding and multi-word union logic natively.
    """
    if not _PADDLE_AVAILABLE or _PADDLE_OCR is None:
        print("    [bbox/ocr] paddleocr not installed — skipping OCR snap", file=sys.stderr)
        return box

    left, top, right, bottom = box
    w, h = img.size

    # Asymmetric padding: tight vertical to stay in row, wide horizontal to catch URLs/long values
    pad_v = max(12, (bottom - top) // 2)
    pad_h = max(150, (right - left) // 2) 

    reg_l = max(0, left   - pad_h)
    reg_t = max(0, top    - pad_v)
    reg_r = min(w, right  + pad_h)
    reg_b = min(h, bottom + pad_v)
    
    region = img.crop((reg_l, reg_t, reg_r, reg_b))
    img_np = np.array(region.convert('RGB')) # PaddleOCR works well with numpy arrays

    # Pre-process recommendation to split on URL delimiters and punctuation
    clean_rec = recommendation
    for char in "?=&+,:":
        clean_rec = clean_rec.replace(char, " ")

    # Extract matchable keywords
    keywords = [
        tok.lower().strip("'\".,;:()[]{}|")
        for tok in clean_rec.split()
        if len(tok.strip("'\".,;:()[]{}|")) >= 2
        and tok.lower().strip("'\".,;:()[]{}|") not in _STOPWORDS
    ]
    
    if not keywords:
        return box

    try:
        # Run PaddleOCR
        result = _PADDLE_OCR.ocr(img_np, cls=False)
    except Exception as exc:
        print(f"    [bbox/ocr] PaddleOCR error: {exc}", file=sys.stderr)
        return box

    if not result or not result[0]:
        return box

    matched_boxes = []
    best_overall_ratio = 0.0

    # Paddle returns: [ [ [x1,y1], [x2,y1], [x2,y2], [x1,y2] ], ('text', confidence) ]
    for line in result[0]:
        coords, (text, confidence) = line
        
        # Skip very low confidence strings
        if confidence < 0.60 or not text.strip():
            continue
            
        # Get bounding box for this text fragment
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        wx1, wy1 = int(min(xs)), int(min(ys))
        wx2, wy2 = int(max(xs)), int(max(ys))

        # Check against keywords
        best_word_ratio = 0.0
        for keyword in keywords:
            # We match against the whole text fragment found by Paddle
            # Paddle often groups words nicely, e.g., "s?k=sd+card" might be one result
            ratio = difflib.SequenceMatcher(None, keyword, text.lower()).ratio()
            best_word_ratio = max(best_word_ratio, ratio)
            
        if best_word_ratio >= _UNION_THRESHOLD:
            best_overall_ratio = max(best_overall_ratio, best_word_ratio)
            matched_boxes.append(
                [reg_l + wx1, reg_t + wy1, reg_l + wx2, reg_t + wy2]
            )

    if best_overall_ratio >= _ACCEPT_THRESHOLD and matched_boxes:
        # Union the matching text regions
        sl = min(b[0] for b in matched_boxes)
        st = min(b[1] for b in matched_boxes)
        sr = max(b[2] for b in matched_boxes)
        sb = max(b[3] for b in matched_boxes)
        # Apply slight margin
        return [max(0, sl - 3), max(0, st - 3), min(w, sr + 3), min(h, sb + 3)]

    return box

# ── Stage 3: OpenCV ────────────────────────────────────────────────────────

def _cluster_lines(positions: list[int], min_gap: int = 4) -> list[int]:
    if not positions: return []
    sorted_pos = sorted(set(positions))
    clusters: list[list[int]] = [[sorted_pos[0]]]
    for p in sorted_pos[1:]:
        if p - clusters[-1][-1] <= min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [round(sum(c) / len(c)) for c in clusters]

def _snap_to_blue_row(img: Image.Image, box: list[int]) -> list[int] | None:
    if not _CV2_AVAILABLE: return None
    left, top, right, bottom = box
    w, h = img.size
    height = max(bottom - top, 10)
    roi_t  = max(0, top    - height)
    roi_b  = min(h, bottom + height)
    region   = img.crop((0, roi_t, w, roi_b))
    arr_rgb  = np.array(region.convert("RGB"), dtype=np.uint8)
    arr_hsv  = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([ 95,  20, 165], dtype=np.uint8)
    upper_blue = np.array([140, 160, 255], dtype=np.uint8)
    mask = cv2.inRange(arr_hsv, lower_blue, upper_blue)
    row_coverage = mask.sum(axis=1) / 255.0
    min_px = max(w * 0.12, 15)
    blue_rows = np.where(row_coverage >= min_px)[0]
    if len(blue_rows) == 0: return None
    bands: list[tuple[int, int]] = []
    bs = be = int(blue_rows[0])
    for r in blue_rows[1:]:
        if int(r) - be <= 3: be = int(r)
        else:
            bands.append((bs, be))
            bs = be = int(r)
    bands.append((bs, be))
    rel_top = top - roi_t
    rel_bot = bottom - roi_t
    best_band: tuple[int, int] | None = None
    best_overlap = -1
    for b_start, b_end in bands:
        overlap = max(0, min(b_end + 1, rel_bot) - max(b_start, rel_top))
        if overlap > best_overlap:
            best_overlap = overlap
            best_band = (b_start, b_end)
    if best_band is None: return None
    abs_top    = roi_t + best_band[0]
    abs_bottom = roi_t + best_band[1] + 1
    if abs_bottom - abs_top < 2: return None
    return [left, abs_top, right, abs_bottom]

def _morphological_line_positions(gray: np.ndarray, rw: int, rh: int) -> tuple[list[int], list[int]]:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_len = max(rw // 4, 20)
    h_k   = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_k)
    h_proj = h_mask.sum(axis=1)
    raw_h  = [int(i) for i in np.where(h_proj > h_len * 255 * 0.25)[0]]
    v_len = max(rh // 4, 20)
    v_k   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_k)
    v_proj = v_mask.sum(axis=0)
    raw_v  = [int(i) for i in np.where(v_proj > v_len * 255 * 0.25)[0]]
    return _cluster_lines(raw_h), _cluster_lines(raw_v)

def _stage3_opencv_refine(img: Image.Image, box: list[int]) -> list[int]:
    if not _CV2_AVAILABLE:
        print("    [bbox/cv2] opencv-python not installed — skipping refinement", file=sys.stderr)
        return box
    left, top, right, bottom = box
    w, h = img.size
    blue_snap = _snap_to_blue_row(img, box)
    if blue_snap is not None:
        print(f"    [bbox/cv2] blue-row snap: y {top}–{bottom} → {blue_snap[1]}–{blue_snap[3]}", file=sys.stderr)
        left, top, right, bottom = blue_snap
        box = blue_snap
    pad = max(20, (right - left) // 2, (bottom - top) // 2)
    rl = max(0, left   - pad)
    rt = max(0, top    - pad)
    rr = min(w, right  + pad)
    rb = min(h, bottom + pad)
    rw = rr - rl
    rh = rb - rt
    region_pil = img.crop((rl, rt, rr, rb))
    img_gray   = np.array(region_pil.convert("L"))
    rel_l, rel_t, rel_r, rel_b = left - rl, top - rt, right - rl, bottom - rt
    h_lines, v_lines = _morphological_line_positions(img_gray, rw, rh)
    if h_lines and v_lines:
        nl = _nearest(v_lines, rel_l, prefer="floor")
        nr = _nearest(v_lines, rel_r, prefer="ceil")
        nt = _nearest(h_lines, rel_t, prefer="floor")
        nb = _nearest(h_lines, rel_b, prefer="ceil")
        if nr > nl and nb > nt:
            return [rl + nl, rt + nt, rl + nr, rt + nb]
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw_h: list[int] = []
    raw_v: list[int] = []
    lines = cv2.HoughLinesP(thresh, rho=1, theta=3.14159265/180, threshold=max(25, rw//4), minLineLength=max(15, rw//5), maxLineGap=6)
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy < 3 and dx > rw // 5: raw_h.append((y1 + y2) // 2)
            elif dx < 3 and dy > rh // 5: raw_v.append((x1 + x2) // 2)
    hough_h = _cluster_lines(raw_h)
    hough_v = _cluster_lines(raw_v)
    if hough_h and hough_v:
        nl = _nearest(hough_v, rel_l, prefer="floor")
        nr = _nearest(hough_v, rel_r, prefer="ceil")
        nt = _nearest(hough_h, rel_t, prefer="floor")
        nb = _nearest(hough_h, rel_b, prefer="ceil")
        if nr > nl and nb > nt:
            return [rl + nl, rt + nt, rl + nr, rt + nb]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    init_area = max(1, (rel_r - rel_l) * (rel_b - rel_t))
    best_iou, best_cnt_box = 0.0, None
    for cnt in contours:
        cx, cy, cw, ch_c = cv2.boundingRect(cnt)
        if cw * ch_c < 16: continue
        cx2, cy2 = cx + cw, cy + ch_c
        ix1, iy1 = max(cx, rel_l), max(cy, rel_t)
        ix2, iy2 = min(cx2, rel_r), min(cy2, rel_b)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = init_area + cw * ch_c - inter
            iou   = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_cnt_box = [rl + cx, rt + cy, rl + cx2, rt + cy2]
    if best_iou > 0.3 and best_cnt_box is not None: return best_cnt_box
    return box

# ── Stage 4: Draw ──────────────────────────────────────────────────────────

def draw_annotations(img: Image.Image, annotations: list[list[int]], color: str = ANNOTATION_COLOR, width: int = ANNOTATION_WIDTH) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for ann in annotations:
        l, t, r, b = int(ann[0]), int(ann[1]), int(ann[2]), int(ann[3])
        if r > l and b > t:
            draw.rectangle([l, t, r, b], outline=color, width=width)
    return img

# ── API ────────────────────────────────────────────────────────────────────

def annotate_image(cropped_img: Image.Image, annotation_recommendations: list[str], q_text: str, region: str | None = None, model_id: str | None = None) -> list[list[int]]:
    if not annotation_recommendations: return []
    client = _get_ann_client(region)
    used_model = model_id or ANNOTATION_MODEL_ID
    w, h = cropped_img.size
    
    nova_results = _stage1_nova_boxes(client, cropped_img, annotation_recommendations, used_model)
    if not nova_results: return []

    refined_boxes: list[list[int]] = []
    for label, box in nova_results[:3]:
        frac = (box[2] - box[0]) * (box[3] - box[1]) / (w * h) if w * h else 0.0
        if not _passes_sanity_gate(box, w, h): continue
        
        # Match label to the closest annotation recommendation for OCR keywords
        best_rec = annotation_recommendations[0]
        best_sim = 0.0
        for rec in annotation_recommendations:
            sim = difflib.SequenceMatcher(None, label.lower(), rec.lower()).ratio()
            if sim > best_sim:
                best_sim = sim
                best_rec = rec
        
        box_ocr = _stage2_paddle_snap(cropped_img, box, best_rec)
        if not _passes_sanity_gate(box_ocr, w, h): box_ocr = box
        
        box_cv = _stage3_opencv_refine(cropped_img, box_ocr)
        if not _passes_sanity_gate(box_cv, w, h): box_cv = box_ocr
        
        refined_boxes.append(box_cv)
        
    return refined_boxes

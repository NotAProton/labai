"""
vision.py — Bedrock analysis pipeline (Step 1 only).

Step 1: Analysis LLM call → crop coordinates + annotation recommendations (text)
Annotation (Stages 2–4) is handled by bounding_box.py.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

from .bounding_box import annotate_image
from .box_adjuster import show_box_adjuster

DEFAULT_REGION = "ap-south-1"

MODEL_ID = "moonshotai.kimi-k2.5"  
REQUEST_DELAY = 1.5

_client = None
_client_region: str | None = None


def _get_client(region: str | None = None):
    """Return a cached Bedrock runtime client for Converse requests."""
    global _client, _client_region
    target_region = region or DEFAULT_REGION
    if _client is None or _client_region != target_region:
        _client = boto3.client("bedrock-runtime", region_name=target_region)
        _client_region = target_region
    return _client


def _image_block(path: Path) -> dict:
    """Return a Bedrock Converse API image content block."""
    fmt = "png" if path.suffix.lower() == ".png" else "jpeg"
    return {"image": {"format": fmt, "source": {"bytes": path.read_bytes()}}}


def _image_block_from_bytes(data: bytes, fmt: str = "png") -> dict:
    """Return a Bedrock Converse API image content block from raw bytes."""
    return {"image": {"format": fmt, "source": {"bytes": data}}}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 SYSTEM PROMPT — Analysis (crop + annotation recommendations as text)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a forensic lab report assistant. Your ONLY job is to analyze
Magnet AXIOM Examine screenshots and produce structured JSON that drives
an automated LaTeX report-generation pipeline. You must follow every rule
below exactly.

───────────────────────────────────────────────────────────────────────────
TOOL REFERENCE: Magnet AXIOM Examine v9.11
───────────────────────────────────────────────────────────────────────────

Magnet AXIOM Examine is a digital forensics platform for analyzing forensic
disk images. Understanding its UI is critical to identifying what to crop,
annotate, and describe.

## UI Layout (standard 3-pane view)
┌─────────────────────────────────────────────────────────────────────────┐
│  TOP BAR: Module tabs (Artifacts / File System / Registry / Timeline) │
│           Global Search box (top-right)                               │
├──────────────────┬──────────────────────────┬──────────────────────────┤
│ LEFT NAV PANE    │ CENTER: EVIDENCE PANE    │ RIGHT: DETAILS PANE      │
│ (artifact tree)  │ (table of artifact rows) │ (field/value pairs)      │
│ - Artifacts      │ Click a row to select    │ Hex view + Data          │
│ - OS             │ column sort/filter       │ Interpreter (bottom)     │
│ - Web Related    │                          │                          │
│ - App Usage      │                          │                          │
└──────────────────┴──────────────────────────┴──────────────────────────┘

## Key Artifact Navigation Paths
| What you want            | AXIOM path                                           |
|--------------------------|------------------------------------------------------|
| User accounts            | Artifacts → Operating System → User Accounts         |
| OS info / install date   | Artifacts → Operating System → Operating System Info |
| Startup programs         | Artifacts → Operating System → Startup Items         |
| Installed software       | Artifacts → Operating System → Installed Programs    |
| DHCP / network leases    | Artifacts → Operating System → Network Interfaces    |
| Prefetch (run count)     | Artifacts → Application Usage → Prefetch Files       |
| Web downloads            | Artifacts → Web Related → Downloads                  |
| Web search terms         | Artifacts → Web Related → Search Terms               |
| Registry keys            | Registry view → key tree on left                     |
| Raw file tree            | File System view                                     |

## Key Field Definitions
- SID: Security Identifier, format S-1-5-21-XXXXXXXX-XXXXXXXX-XXXXXXXX-RID
- RID: Relative Identifier = the last number in the SID (e.g. 1001, 1002)
- F value: Binary SAM registry value; first 8 bytes = Last Login in Windows
  FILETIME (100-ns intervals since 1601-01-01 UTC)
- Windows FILETIME: 64-bit little-endian integer timestamp
- Data Interpreter: panel at bottom-right of Details Pane; shows decoded
  values of highlighted hex bytes (e.g. "Windows 64-bit Hex LE")
- /background arg in startup items means the program runs in the background

## Case Facts (hard-coded for reference — do NOT contradict these)
- Forensic image: Windows 10 machine
- Local users: ryanJ (RID 1001) and RJennings (RID 1002)
- Email / internet username: ryanJennings1842@outlook.com
- OS install date: 12-01-2022 13:54:15
- Timezone: UTC-05:00 Indiana (East)
- ryanJ login count: 0  |  RJennings login count: 6
- RJennings last bad login: 12-09-2022 06:11:12.000
- Signal version: 5.58.0
- TOR run count: 2  |  TOR last run: 10-10-2022 10:08:18.999
- DHCP lease for 172.11.92.147: 4 hours
- OneDrive startup configured with /background argument → True

───────────────────────────────────────────────────────────────────────────
COORDINATE SYSTEM & BOUNDING BOX RULES
───────────────────────────────────────────────────────────────────────────

All x/y coordinates you return are FRACTIONS of the ORIGINAL image
dimensions (width W and height H), in the range [0.0, 1.0].

"crop" trims the image to a sub-region of the original.

CRITICAL CROP RULES:
1. RETAIN UI CONTEXT: Never crop so tightly that you lose the navigational
   context. You MUST include column headers, pane titles, or the selected
   artifact row in the Evidence Pane so the user knows where the data lives.
2. TRIM DEAD SPACE: Always crop out the Windows taskbar, the main application
   title bar, and large areas of empty white space.
3. ALMOST ALWAYS CROP: You must apply a crop to focus the viewer. Do NOT
   return `null` for a crop unless the evidence physically spans from the
   absolute top-left to the bottom-right of the monitor.

───────────────────────────────────────────────────────────────────────────
ANNOTATION RECOMMENDATIONS (TEXT, NOT COORDINATES)
───────────────────────────────────────────────────────────────────────────

Instead of providing annotation coordinates, describe IN TEXT what should
be annotated. A separate annotation step will use the cropped image to
produce precise bounding boxes.

Rules for annotation recommendations:
1. Be specific: name the exact text, value, cell, or field to highlight.
2. Maximum 3 recommendations per image.
3. Describe the location within the AXIOM UI (e.g., "the 'Login Count'
   value '6' in the Details Pane", "the selected row for 'RJennings' in
   the Evidence Pane").

───────────────────────────────────────────────────────────────────────────
AVAILABLE LATEX MACROS (use these in explanation and answer_latex)
───────────────────────────────────────────────────────────────────────────

\\ans{value}           — bold, accent-colored inline answer value
\\ansbox{sentence}     — highlighted answer box (full sentence with \\ans{})
\\texttt{text}         — monospace font for filenames, paths, registry keys,
                         usernames, commands
\\newline              — line break inside \\ansbox{} when needed
\\textbackslash{}      — literal backslash character in text

Escape rules for LaTeX text:
  & → \\&    % → \\%    # → \\#    _ → \\_    $ → \\$
  Do NOT use raw \\ for backslashes in paths; use \\textbackslash{}



───────────────────────────────────────────────────────────────────────────
OUTPUT JSON SCHEMA — return EXACTLY this structure, nothing else
───────────────────────────────────────────────────────────────────────────

{
  "question_summary": "Summarize the question and what to find in one or two sentences.",
  "images": [
    {
      "source": "image-01.png",
      "crop": [left, top, right, bottom],
      "annotation_recommendations": [
        "Describe what to annotate — e.g. 'The Login Count value 6 in the Details Pane next to the field label'",
        "The selected row for RJennings in the Evidence Pane center table"
      ],
      "caption": "Axiom Examine v9.11: ...",
      "output_name": "01.png"
    }
  ],
  "explanation": "...",
  "answer_latex": "..."
}

## Field rules

"question_summary"
  A concise summary of the question and what to find, in one or two sentences. For multipart questions,
  include all parts in the summary in a second sentence. 

"source"
  The original filename exactly as given to you.

"crop"
  [left, top, right, bottom] fractions of the original image. You MUST
  return a 4-element array trimming dead space (taskbars, blank areas)
  while keeping AXIOM navigation headers visible. Do NOT use null unless
  entirely unavoidable.

"annotation_recommendations"
  Array of 1–3 strings. Each string is a natural-language description of
  what to annotate on the cropped image. Be specific about the text/value
  and its location in the UI, so the annotator AI can find it.

"caption"
  Must start with "Axiom Examine v9.11: " or whichever program is being
  used. Describe the artifact view and what is highlighted.

"output_name"
  Strip the "image-" prefix from the source filename:
  "image-01.png" → "01.png". Preserve unrelated prefixes.

"explanation"
  1 to 3 sentences in first-person plural. Describe the AXIOM navigation
  path and what the screenshot reveals.

"answer_latex"
  Content for the \\ansbox{} macro. Write a complete grammatical sentence.
  Wrap key answer values in \\ans{}. Use \\mono{} for usernames, paths,
  filenames, registry keys, and commands.

───────────────────────────────────────────────────────────────────────────
ABSOLUTE RULES
───────────────────────────────────────────────────────────────────────────

1. Return ONLY valid JSON. No markdown fences. No extra text before or after
   the JSON object.
2. All coordinate values must be floats in [0.0, 1.0].
3. "crop" must be null or a 4-element array.
4. When screenshots are provided, include one image entry per screenshot.
5. Never contradict the Case Facts listed above.
6. JSON ESCAPING (CRITICAL): every backslash inside a JSON string value MUST
   be doubled. Write \\\\mono{}, \\\\ans{}, \\\\newline,
   \\\\textbackslash{} — NOT \\mono{}, \\ans{}, etc.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Pre-annotated images mode (--skip-modify-images)
# Images are already cropped + annotated with red boxes; no further image
# processing will occur.  Qwen must use the red annotations as primary evidence.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PREANNOTATED = """\
You are a forensic lab report assistant. Your ONLY job is to analyze
Magnet AXIOM Examine screenshots and produce structured JSON that drives
an automated LaTeX report-generation pipeline. You must follow every rule
below exactly.

───────────────────────────────────────────────────────────────────────────
PRE-ANNOTATED IMAGES — READ THIS FIRST
───────────────────────────────────────────────────────────────────────────

The screenshots you receive have ALREADY been fully processed:
  • They are already CROPPED to the relevant region.
  • They already contain RED BOUNDING BOXES drawn around the key evidence.

Your PRIMARY guide is the red bounding box(es) in each screenshot.
Read what is inside or adjacent to each red box first.  If red boxes are
absent or ambiguous, fall back to the most visually obvious evidence in
the screenshot (e.g., the selected/highlighted row, the topmost result,
the largest value in the field).

Use this priority order when identifying the answer:
  1. Content highlighted by a RED bounding box.
  2. The most obvious, unambiguous piece of evidence in the screenshot.

───────────────────────────────────────────────────────────────────────────
TOOL REFERENCE: Magnet AXIOM Examine v9.11
───────────────────────────────────────────────────────────────────────────

Magnet AXIOM Examine is a digital forensics platform for analyzing forensic
disk images. Understanding its UI is critical to identifying what to crop,
annotate, and describe.

## UI Layout (standard 3-pane view)
┌─────────────────────────────────────────────────────────────────────────┐
│  TOP BAR: Module tabs (Artifacts / File System / Registry / Timeline) │
│           Global Search box (top-right)                               │
├──────────────────┬──────────────────────────┬──────────────────────────┤
│ LEFT NAV PANE    │ CENTER: EVIDENCE PANE    │ RIGHT: DETAILS PANE      │
│ (artifact tree)  │ (table of artifact rows) │ (field/value pairs)      │
│ - Artifacts      │ Click a row to select    │ Hex view + Data          │
│ - OS             │ column sort/filter       │ Interpreter (bottom)     │
│ - Web Related    │                          │                          │
│ - App Usage      │                          │                          │
└──────────────────┴──────────────────────────┴──────────────────────────┘

## Key Artifact Navigation Paths
| What you want            | AXIOM path                                           |
|--------------------------|------------------------------------------------------|
| User accounts            | Artifacts → Operating System → User Accounts         |
| OS info / install date   | Artifacts → Operating System → Operating System Info |
| Startup programs         | Artifacts → Operating System → Startup Items         |
| Installed software       | Artifacts → Operating System → Installed Programs    |
| DHCP / network leases    | Artifacts → Operating System → Network Interfaces    |
| Prefetch (run count)     | Artifacts → Application Usage → Prefetch Files       |
| Web downloads            | Artifacts → Web Related → Downloads                  |
| Web search terms         | Artifacts → Web Related → Search Terms               |
| Registry keys            | Registry view → key tree on left                     |
| Raw file tree            | File System view                                     |

## Key Field Definitions
- SID: Security Identifier, format S-1-5-21-XXXXXXXX-XXXXXXXX-XXXXXXXX-RID
- RID: Relative Identifier = the last number in the SID (e.g. 1001, 1002)
- F value: Binary SAM registry value; first 8 bytes = Last Login in Windows
  FILETIME (100-ns intervals since 1601-01-01 UTC)
- Windows FILETIME: 64-bit little-endian integer timestamp
- Data Interpreter: panel at bottom-right of Details Pane; shows decoded
  values of highlighted hex bytes (e.g. "Windows 64-bit Hex LE")
- /background arg in startup items means the program runs in the background

## Case Facts (hard-coded for reference — do NOT contradict these)
- Forensic image: Windows 10 machine
- Local users: ryanJ (RID 1001) and RJennings (RID 1002)
- Email / internet username: ryanJennings1842@outlook.com
- OS install date: 12-01-2022 13:54:15
- Timezone: UTC-05:00 Indiana (East)
- ryanJ login count: 0  |  RJennings login count: 6
- RJennings last bad login: 12-09-2022 06:11:12.000
- Signal version: 5.58.0
- TOR run count: 2  |  TOR last run: 10-10-2022 10:08:18.999
- DHCP lease for 172.11.92.147: 4 hours
- OneDrive startup configured with /background argument → True

───────────────────────────────────────────────────────────────────────────
AVAILABLE LATEX MACROS (use these in explanation and answer_latex)
───────────────────────────────────────────────────────────────────────────

\\ans{value}           — bold, accent-colored inline answer value
\\ansbox{sentence}     — highlighted answer box (full sentence with \\ans{})
\\texttt{text}         — monospace font for filenames, paths, registry keys,
                         usernames, commands
\\newline              — line break inside \\ansbox{} when needed
\\textbackslash{}      — literal backslash character in text

Escape rules for LaTeX text:
  & → \\&    % → \\%    # → \\#    _ → \\_    $ → \\$
  Do NOT use raw \\ for backslashes in paths; use \\textbackslash{}

───────────────────────────────────────────────────────────────────────────
OUTPUT JSON SCHEMA — return EXACTLY this structure, nothing else
───────────────────────────────────────────────────────────────────────────

{
  "question_summary": "Summarize the question and what to find in one or two sentences.",
  "images": [
    {
      "source": "01.png",
      "crop": null,
      "annotation_recommendations": [],
      "caption": "Axiom Examine v9.11: ...",
      "output_name": "01.png"
    }
  ],
  "explanation": "...",
  "answer_latex": "..."
}

## Field rules

"question_summary"
  A concise summary of the question and what to find, in one or two sentences. For multipart questions,
  include all parts in the summary in a second sentence.

"source"
  The original filename exactly as given to you.

"crop"
  MUST be null — the image is already cropped. Do not return coordinates.

"annotation_recommendations"
  MUST be an empty array []. Annotation has already been applied.

"caption"
  Must start with "Axiom Examine v9.11: " or whichever program is being
  used. Describe the artifact view and what is highlighted (including
  what the red box surrounds, if visible).

"output_name"
  Use the same filename as "source" — do not modify it.

"explanation"
  1 to 3 sentences in first-person plural. Describe the AXIOM navigation
  path and what the screenshot (and its red annotation box) reveals.

"answer_latex"
  Content for the \\ansbox{} macro. Write a complete grammatical sentence.
  Wrap key answer values in \\ans{}. Use \\mono{} for usernames, paths,
  filenames, registry keys, and commands.

───────────────────────────────────────────────────────────────────────────
ABSOLUTE RULES
───────────────────────────────────────────────────────────────────────────

1. Return ONLY valid JSON. No markdown fences. No extra text before or after
   the JSON object.
2. "crop" must always be null in this mode.
3. "annotation_recommendations" must always be [] in this mode.
4. When screenshots are provided, include one image entry per screenshot.
5. Never contradict the Case Facts listed above.
6. JSON ESCAPING (CRITICAL): every backslash inside a JSON string value MUST
   be doubled. Write \\\\mono{}, \\\\ans{}, \\\\newline,
   \\\\textbackslash{} — NOT \\mono{}, \\ans{}, etc.
"""



def _parse_json(raw: str) -> dict:
    """Strip markdown fences, repair bare LaTeX backslashes, and parse JSON."""
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\\(?=[a-zA-Z])', r'\\\\', raw)
        return json.loads(fixed)


# ─────────────────────────────────────────────────────────────────────────────
# PIL helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_image(image_path: Path, crop_frac: list[float]) -> Image.Image:
    """
    Crop an image given fractional [left, top, right, bottom] coordinates.
    Returns the cropped PIL Image.
    """
    img = Image.open(image_path)
    w, h = img.size
    left = int(crop_frac[0] * w)
    top = int(crop_frac[1] * h)
    right = int(crop_frac[2] * w)
    bottom = int(crop_frac[3] * h)
    # Clamp to image bounds
    left = max(0, min(left, w))
    top = max(0, min(top, h))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return img.crop((left, top, right, bottom))



# ────────────────────────────────────────────────────────────────────────��────
# LLM call helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_converse(client, system_prompt: str, content: list[dict],
                   max_tokens: int = 2048, temperature: float = 0.1) -> str:
    """Make a single Bedrock Converse call and return the raw text response."""
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        serviceTier={'type': 'flex'},
    )
    time.sleep(REQUEST_DELAY)
    return "\n".join(
        block["text"]
        for block in response["output"]["message"].get("content", [])
        if "text" in block
    ).strip()


def _step1_analyze(
    client,
    module_title: str,
    module_num: int,
    q_num: int,
    q_text: str,
    q_context: str,
    image_paths: list[Path],
    nav_hint: str,
    preannotated: bool = False,
) -> dict:
    """
    Step 1: Send screenshots to LLM for analysis.
    When *preannotated* is True, uses SYSTEM_PROMPT_PREANNOTATED and tells the
    model that images are already cropped/annotated with red boxes.
    Returns dict with (possibly null) crop coords and annotation recommendations.
    """
    if preannotated:
        system_prompt = SYSTEM_PROMPT_PREANNOTATED
        extra_instruction = (
            "The images have already been cropped and annotated with RED bounding boxes "
            "highlighting the key evidence. Use the red boxes as your PRIMARY guide. "
            "Set \"crop\" to null and \"annotation_recommendations\" to [] for every image."
        )
    else:
        system_prompt = SYSTEM_PROMPT
        extra_instruction = (
            "Carefully choose crop regions and write clear annotation recommendations "
            "describing what should be highlighted."
        )

    user_prompt = f"""\
Analyze this forensic lab question and its associated screenshot(s).

## Question
Module {module_num} — {module_title}
Q{module_num}.{q_num}: {q_text}

## Answer Notes (from the lab notebook — use these as authoritative)
{q_context.strip() if q_context.strip() else "(No explicit answer provided — infer it from the screenshot if possible.)"}

## AXIOM Navigation Steps (from the lab walkthrough)
{nav_hint}

## Provided Screenshots (listed in the same order as the image parts)
{[p.name for p in image_paths] if image_paths else "(no screenshots)"}

Produce the JSON output exactly as specified in the system prompt.
{extra_instruction}
"""

    content: list[dict] = [_image_block(path) for path in image_paths]
    content.append({"text": user_prompt})

    last_error: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(1.0)
        try:
            raw = _call_converse(client, system_prompt, content)
        except (ClientError, BotoCoreError) as exc:
            print(
                f"    [bedrock] Step 1 call failed (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            last_error = exc
            continue

        try:
            result = _parse_json(raw)
        except json.JSONDecodeError as exc:
            print(
                f"    [json] Step 1 invalid JSON (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            print(f"    Raw (first 600 chars): {raw[:600]}", file=sys.stderr)
            last_error = exc
            continue

        required = {"question_summary", "images", "explanation", "answer_latex"}
        missing = required - result.keys()
        if not missing:
            return result

        print(
            f"    [json] Step 1 missing keys {missing} (attempt {attempt + 1}/3), retrying…",
            file=sys.stderr,
        )
        last_error = ValueError(f"missing keys: {missing}")

    raise RuntimeError(
        f"Step 1 failed after 3 attempts for Q{module_num}.{q_num}: {last_error}"
    )



# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_question(
    module_title: str,
    module_num: int,
    q_num: int,
    q_text: str,
    q_context: str,
    image_paths: list[Path],
    nav_hint: str,
    region: str | None = None,
    annotation_region: str | None = None,
    annotation_model: str | None = None,
    allow_manual_adjustment: bool = True,
    skip_modify_images: bool = False,
) -> dict:
    """
    Analysis pipeline for a forensic lab question.

    Step 1: LLM (Qwen via Bedrock) analyzes original images → crop + annotation
            recommendations (text).  Region / model controlled by *region*.
    Stages 2–4: Delegated to bounding_box.annotate_image() — Amazon Nova LLM
            grounding → Tesseract OCR snap → OpenCV Hough-line refinement →
            PIL drawing.  Region / model controlled by *annotation_region* /
            *annotation_model*.
    Stage 5:  Manual adjustment GUI (Tkinter) — pauses the pipeline and lets
            the user move, resize, delete, or create bounding boxes before
            the result is finalised.  Skipped when *allow_manual_adjustment*
            is False (e.g. ``--skip-manual-adjustment`` CLI flag).

    When *skip_modify_images* is True (``--skip-modify-images`` CLI flag):
      - *image_paths* must already point to the processed images in
        output-images (e.g. work/lab7img/01.png).
      - The pre-annotated system prompt is used so Qwen reads the red boxes.
      - Stages 2–5 (crop / Nova bbox / manual GUI) are skipped entirely.
      - The output-images folder is not touched.

    Returns a dict with keys: question_summary, images, explanation, answer_latex.
    Each image entry includes final annotation coordinates (pixel coords in
    cropped-image space) that annotator.py uses for drawing.
    """
    client = _get_client(region)

    # ── Step 1: Analysis ────────────────────────────────────────────────────
    mode_tag = "pre-annotated" if skip_modify_images else "standard"
    print(f"    [step 1] Analyzing Q{module_num}.{q_num} ({mode_tag})…", file=sys.stderr)
    step1_result = _step1_analyze(
        client, module_title, module_num, q_num,
        q_text, q_context, image_paths, nav_hint,
        preannotated=skip_modify_images,
    )
    print(f"[step 1] Analysis complete: {step1_result['question_summary']}")

    # ── Skip image processing when --skip-modify-images ─────────────────────
    if skip_modify_images:
        for img_entry in step1_result.get("images", []):
            img_entry["annotations"] = []
            img_entry.pop("annotation_recommendations", None)
        return step1_result

    # ── Step 2: Crop + Annotate each image ──────────────────────────────────
    path_lookup = {p.name: p for p in image_paths}

    for img_entry in step1_result.get("images", []):
        source = img_entry.get("source", "")
        crop_coords = img_entry.get("crop")
        ann_recs = img_entry.get("annotation_recommendations", [])

        img_path = path_lookup.get(source)
        if img_path is None or not img_path.exists():
            print(
                f"    [warn] Source image '{source}' not found, skipping annotation step",
                file=sys.stderr,
            )
            img_entry["annotations"] = []
            continue

        # Crop the image
        if crop_coords and len(crop_coords) == 4:
            print(f"    [step 2] Cropping {source} → {crop_coords}", file=sys.stderr)
            cropped = crop_image(img_path, crop_coords)
        else:
            print(f"    [step 2] No crop for {source}, using full image", file=sys.stderr)
            cropped = Image.open(img_path)

        # Stages 2–4: Nova LLM grounding → OCR snap → OpenCV refinement
        if ann_recs:
            print(
                f"    [bbox] Annotating {source} ({len(ann_recs)} recommendation(s))…",
                file=sys.stderr,
            )
            annotations = annotate_image(
                cropped, ann_recs, q_text,
                region=annotation_region,
                model_id=annotation_model,
            )
        else:
            annotations = []

        # Stage 5: Manual bounding-box adjustment GUI
        if allow_manual_adjustment and (annotations or ann_recs):
            print(
                f"    [bbox] Waiting for manual adjustment of {source}…",
                file=sys.stderr,
            )
            annotations = show_box_adjuster(cropped, annotations, ann_recs)
            print(
                f"    [bbox] Manual adjustment confirmed: {len(annotations)} box(es)",
                file=sys.stderr,
            )

        img_entry["annotations"] = annotations
        img_entry.pop("annotation_recommendations", None)
        img_entry["_cropped_pil"] = cropped

    return step1_result

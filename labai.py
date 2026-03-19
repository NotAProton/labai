#!/usr/bin/env python3
"""
Lab Report Generator
--------------------
Extracts page screenshots from a PDF, transcribes each via Gemini,
then generates a structured Markdown lab report.

Usage:
    python process_lab.py <path_to_pdf>

Output:
    screens/          - one PNG per PDF page
    transcriptions/   - one .txt per page
    lab_report.md     - final formatted report
"""

import sys
import os
import json
import time
import base64
import re
from io import BytesIO
from pathlib import Path
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = "AIzaSyD1ZvQa1JOjBng0rLoFDxCLTfr_m9gQ_p0"
MODEL = "gemini-3.1-flash-lite-preview"
DPI     = 150                          # resolution for page renders
DELAY   = 1.5                         # seconds between API calls (rate-limit safety)

# ── Prompts ─────────────────────────────────────────────────────────────────
TRANSCRIBE_PROMPT = """You are helping document a cloud computing lab assignment.
This image is one screenshot from a step-by-step lab walkthrough done on AWS Console,
a local bash terminal SSH'd into an EC2 instance, or a similar tool.

Carefully describe what is shown in this screenshot:
1. What application/interface is visible? (AWS Console page/service, terminal, browser, etc.)
2. What action was just performed or is being shown?
3. What are the key visible details? (resource names, commands typed, outputs, settings, button clicked, form filled, etc.)
4. Is there a caption or label on the image? If so, quote it exactly.

Be precise and thorough — this transcription will be used to reconstruct lab instructions.
Format your response as plain text, no markdown."""

STRUCTURE_PROMPT = """You are analyzing transcriptions of screenshots from a cloud computing lab assignment (AWS).
The screenshots are in order. Each transcription corresponds to one screenshot/step.

Here are all the transcriptions:
{transcriptions}

Based on these transcriptions:
1. Identify how many distinct tasks/sections the lab has (look for logical groupings like "launching an EC2 instance", "configuring security groups", "SSH into instance", etc.)
2. For each task, identify which screenshot numbers (1-based) belong to it.
3. Give each task a concise name (e.g., "Launch EC2 Instance", "Configure Security Groups").

Respond ONLY with valid JSON (no markdown, no backticks), in this exact format:
{{
  "tasks": [
    {{
      "name": "Task Name Here",
      "description": "One sentence describing what this task accomplishes.",
      "screenshot_range": [1, 5]
    }}
  ]
}}"""

REPORT_PROMPT = """You are writing a professional lab report for a cloud computing lab assignment.

Lab structure (tasks and screenshot ranges):
{structure}

Screenshot transcriptions (numbered):
{transcriptions}

Write a Markdown lab report following this EXACT format — do not deviate:

# Cloud Computing Lab 7

## Overview
[2-3 sentence summary of what the entire lab accomplished]

---

[For each task, use this format:]

## [Task Name]
### [One sentence describing what this task accomplishes]

[For each step in the task:]
### Step [N]: [Action in imperative style — just the action, no explanation. Examples: "Navigate to EC2 Dashboard", "Click Launch Instance", "Run the ssh command in terminal"]
![Step [N] - Brief description](screens/page_[ZERO_PADDED_PAGE_NUMBER].png)

---

Rules:
- Steps must be in imperative style: start with a verb ("Click", "Enter", "Navigate", "Run", "Select", "Copy", "Open", "Set", "Enable")
- One step per screenshot
- Screenshot filename format: screens/page_001.png, screens/page_002.png, etc. (zero-pad to 3 digits matching the page number)
- Do not add reasoning or explanation to steps — just the action
- Keep task descriptions to one sentence
- Include every screenshot as a step"""

# ── Helpers ─────────────────────────────────────────────────────────────────

def setup_dirs(base: Path):
    screens = base / "screens"
    txts    = base / "transcriptions"
    screens.mkdir(exist_ok=True)
    txts.mkdir(exist_ok=True)
    return screens, txts


def extract_pages_as_images(pdf_path: Path, screens_dir: Path) -> list[Path]:
    """Extract all embedded images from the PDF in order of occurrence."""
    doc   = fitz.open(str(pdf_path))
    pages = []
    idx   = 1
    seen  = set()  # deduplicate by xref (same resource referenced on multiple pages)

    for page in doc:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen:
                continue
            seen.add(xref)
            out = screens_dir / f"page_{str(idx).zfill(3)}.png"
            if out.exists():
                print(f"  [skip] {out.name} already exists")
            else:
                img_data = doc.extract_image(xref)
                raw, ext = img_data["image"], img_data["ext"]
                if ext.lower() == "png":
                    out.write_bytes(raw)
                else:
                    Image.open(BytesIO(raw)).save(str(out), "PNG")
                print(f"  Saved {out.name}")
            pages.append(out)
            idx += 1

    print(f"  Extracted {len(pages)} images.")
    doc.close()
    return pages


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def transcribe_image(model, img_path: Path, page_num: int) -> str:
    """Send one image to Gemini and return the transcription text."""
    img   = Image.open(img_path)
    response = model.generate_content(
        [TRANSCRIBE_PROMPT, img],
        generation_config={"temperature": 0.2}
    )
    return response.text.strip()


def transcribe_all(model, pages: list[Path], txts_dir: Path) -> list[str]:
    """Transcribe each page; cache results to disk."""
    transcriptions = []
    for i, img_path in enumerate(pages, start=1):
        cache = txts_dir / f"page_{str(i).zfill(3)}.txt"
        if cache.exists():
            print(f"  [cache] {cache.name}")
            text = cache.read_text(encoding="utf-8")
        else:
            print(f"  Transcribing page {i}/{len(pages)} …", end=" ", flush=True)
            text = transcribe_image(model, img_path, i)
            cache.write_text(text, encoding="utf-8")
            print("done")
            time.sleep(DELAY)
        transcriptions.append(text)
    return transcriptions


def build_numbered_block(transcriptions: list[str]) -> str:
    lines = []
    for i, t in enumerate(transcriptions, start=1):
        lines.append(f"=== Screenshot {i} ===\n{t}\n")
    return "\n".join(lines)


def parse_structure(model, transcriptions: list[str]) -> dict:
    numbered = build_numbered_block(transcriptions)
    prompt   = STRUCTURE_PROMPT.format(transcriptions=numbered)
    print("  Asking Gemini to identify tasks …", end=" ", flush=True)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1}
    )
    raw = resp.text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    print("done")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n  WARNING: Could not parse structure JSON: {e}")
        print(f"  Raw response:\n{raw}\n")
        # Fallback: treat everything as one task
        return {
            "tasks": [
                {
                    "name": "Lab Tasks",
                    "description": "Complete lab walkthrough.",
                    "screenshot_range": [1, len(transcriptions)]
                }
            ]
        }


def generate_report(model, structure: dict, transcriptions: list[str]) -> str:
    numbered = build_numbered_block(transcriptions)
    prompt   = REPORT_PROMPT.format(
        structure=json.dumps(structure, indent=2),
        transcriptions=numbered
    )
    print("  Generating final lab report …", end=" ", flush=True)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 8192}
    )
    print("done")
    return resp.text.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_lab.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    base_dir = pdf_path.parent
    screens_dir, txts_dir = setup_dirs(base_dir)
    report_path = base_dir / "lab_report.md"

    # ── Configure Gemini ───────────────────────────────────────────────────
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL)
    print(f"\n✓ Gemini model: {MODEL}")

    # ── Step 1: Extract pages ──────────────────────────────────────────────
    print("\n[1/4] Extracting PDF pages as images …")
    pages = extract_pages_as_images(pdf_path, screens_dir)
    print(f"  → {len(pages)} images saved to {screens_dir}/")

    # ── Step 2: Transcribe ─────────────────────────────────────────────────
    print(f"\n[2/4] Transcribing {len(pages)} screenshots with Gemini …")
    transcriptions = transcribe_all(model, pages, txts_dir)
    print(f"  → Transcriptions cached in {txts_dir}/")

    # ── Step 3: Identify structure ─────────────────────────────────────────
    print("\n[3/4] Analysing lab structure …")
    time.sleep(DELAY)
    structure = parse_structure(model, transcriptions)
    n_tasks = len(structure.get("tasks", []))
    print(f"  → Found {n_tasks} task(s):")
    for t in structure.get("tasks", []):
        r = t.get("screenshot_range", [])
        print(f"      • {t['name']}  (screenshots {r[0]}–{r[1]})")

    # ── Step 4: Generate report ────────────────────────────────────────────
    print("\n[4/4] Writing lab report …")
    time.sleep(DELAY)
    report_md = generate_report(model, structure, transcriptions)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  → Report saved to {report_path}")

    print("\n✅ All done!")
    print(f"   screens/       → {len(pages)} PNG screenshots")
    print(f"   transcriptions/ → {len(pages)} cached transcriptions")
    print(f"   lab_report.md  → final Markdown report")


if __name__ == "__main__":
    main()

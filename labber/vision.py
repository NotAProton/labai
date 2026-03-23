"""
vision.py — Kimi K2.5 image analysis via AWS Bedrock.

Each call sends a forensic lab question plus all its screenshots to
Kimi K2.5 and gets back structured crop/annotation/caption/text data
as JSON.
"""
import json
import re
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

DEFAULT_REGION = "ap-south-1"
MODEL_ID = "moonshotai.kimi-k2.5"

# ---------------------------------------------------------------------------
# System prompt — comprehensive context for Kimi K2.5.
# Kimi needs more explicit instructions than Claude, so we over-specify
# everything: UI layout, artifact glossary, coordinate system, JSON schema.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a forensic lab report assistant. Your ONLY job is to analyze \
Magnet AXIOM Examine screenshots and produce structured JSON that drives \
an automated LaTeX report-generation pipeline. You must follow every rule \
below exactly.

───────────────────────────────────────────────────────────────────────────
TOOL REFERENCE: Magnet AXIOM Examine v9.11
───────────────────────────────────────────────────────────────────────────

Magnet AXIOM Examine is a digital forensics platform for analyzing forensic
disk images. Understanding its UI is critical to identifying what to crop
and annotate.

## UI Layout (standard 3-pane view)
┌─────────────────────────────────────────────────────────────────────────┐
│  TOP BAR: Module tabs (Artifacts / File System / Registry / Timeline)    │
│           Global Search box (top-right)                                  │
├──────────────────┬──────────────────────────┬───────────────────────────┤
│ LEFT NAV PANE    │  CENTER: EVIDENCE PANE   │  RIGHT: DETAILS PANE      │
│ (artifact tree)  │  (table of artifact rows)│  (field: value pairs for  │
│ - Artifacts      │  Click a row to select   │   selected row)           │
│   - OS           │  column sort/filter      │  Hex view + Data          │
│   - Web Related  │                          │  Interpreter (bottom)     │
│   - App Usage    │                          │                           │
└──────────────────┴──────────────────────────┴───────────────────────────┘

## Key Artifact Navigation Paths
| What you want            | AXIOM path                                           |
|--------------------------|------------------------------------------------------|
| User accounts            | Artifacts → Operating System → User Accounts         |
| OS info / install date   | Artifacts → Operating System → Operating System Info |
| Startup programs         | Artifacts → Operating System → Startup Items         |
| Installed software       | Artifacts → Operating System → Installed Programs    |
| DHCP / network leases    | Artifacts → Operating System → Network Interfaces   |
| Prefetch (run count)     | Artifacts → Application Usage → Prefetch Files      |
| Web downloads            | Artifacts → Web Related → Downloads                 |
| Web search terms         | Artifacts → Web Related → Search Terms              |
| Registry keys            | Registry view → key tree on left                    |
| Raw file tree            | File System view                                    |

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
COORDINATE SYSTEM
───────────────────────────────────────────────────────────────────────────

All x/y coordinates you return are FRACTIONS of the ORIGINAL image
dimensions (width W and height H), in the range [0.0 … 1.0].

  x=0.0, y=0.0  ← top-left corner of the original image
  x=1.0, y=1.0  ← bottom-right corner of the original image

"crop" trims the image to a sub-region of the original.
"annotations" are drawn on the cropped image, but their coordinates
are still expressed in the ORIGINAL image's coordinate space.
The annotator pipeline subtracts the crop origin automatically.

Example: if crop=[0.5, 0.0, 1.0, 0.5] (right half, top half),
and you annotate [0.6, 0.1, 0.9, 0.4], the rectangle will appear
at relative position (0.1, 0.1)→(0.4, 0.4) inside the cropped image.

───────────────────────────────────────────────────────────────────────────
AVAILABLE LATEX MACROS (use these in answer_latex and explanation)
───────────────────────────────────────────────────────────────────────────

\\ans{value}           — bold, accent-colored inline answer value
\\ansbox{sentence}     — highlighted answer box (full sentence with \\ans{})
\\texttt{text}         — monospace font for filenames, paths, registry keys,
                         usernames, commands
\\newline              — line break inside \\ansbox{} when needed
\\textbackslash{}      — literal backslash character in text

Escape rules for LaTeX text:
  &  →  \\&    %  →  \\%    #  →  \\#    _  →  \\_    $  →  \\$
  Do NOT use raw \\ for backslashes in paths; use \\textbackslash{}

───────────────────────────────────────────────────────────────────────────
OUTPUT JSON SCHEMA — return EXACTLY this structure, nothing else
───────────────────────────────────────────────────────────────────────────

{
  "images": [
    {
      "source": "image-01.png",
      "crop": [left, top, right, bottom],
      "annotations": [
        [left, top, right, bottom]
      ],
      "caption": "Axiom Examine v9.11: …",
      "output_name": "01.png"
    }
  ],
  "explanation": "…",
  "answer_latex": "…"
}

## Field rules

"source"
  The original filename exactly as given to you (e.g. "image-01.png").

"crop"
  [left, top, right, bottom] fractions of original. Crop TIGHTLY around
  the region that answers the question — usually the Details Pane or the
  specific Evidence Pane rows. Leave ~2% margin. Use null ONLY if the full
  image is needed to show navigation context (e.g. showing the left nav
  tree in addition to the content).

"annotations"
  Array of [left, top, right, bottom] arrays (ORIGINAL image fractions).
  Draw red rectangles around the exact field name+value row that answers
  the question. One rectangle per key piece of evidence. Empty array [] if
  the crop already isolates the answer tightly enough.

"caption"
  Must start with "Axiom Examine v9.11: ". Describe the specific artifact
  view and what is highlighted. Be concrete:
  GOOD: "Axiom Examine v9.11: User Accounts artifact showing RJennings
         Login Count field highlighted as 6"
  BAD:  "Axiom Examine v9.11: Screenshot of AXIOM"

"output_name"
  Strip the "image-" prefix from the source filename:
  "image-01.png" → "01.png",  "image-23.png" → "23.png"
  Preserve any other prefix (e.g. "extra-01.png" stays "extra-01.png").

"explanation"
  1–3 sentences in first-person plural. Describe the AXIOM navigation path
  and what the screenshot reveals. End naturally before the answer.
  Example: "We navigate to Artifacts → Operating System → User Accounts
  and select the \\texttt{RJennings} account. The Details Pane shows the
  Login Count field set to 6."

"answer_latex"
  Content for the \\ansbox{} macro — a complete grammatical sentence.
  Wrap every key answer value in \\ans{}. Use \\texttt{} for code/paths.
  Example: "The account \\texttt{ryanJ} has logged in \\ans{0} times,
  while \\texttt{RJennings} has logged in \\ans{6} times."

───────────────────────────────────────────────────────────────────────────
ABSOLUTE RULES
───────────────────────────────────────────────────────────────────────────

1. Return ONLY valid JSON. No markdown fences (```). No text before or
   after the JSON object. Start your response with { and end with }.
2. All coordinate values must be floats in [0.0, 1.0].
3. "crop" must be null or a 4-element array.
4. If a question has no screenshots (images list is empty), set
   "images": [] and write text-only explanation and answer_latex.
5. When multiple screenshots are provided, include one entry per image.
   Each screenshot may show a different step; describe each one.
6. explanation and answer_latex must be LaTeX-safe (escape special chars).
7. Never contradict the Case Facts listed above.
"""


def _image_block(path: Path) -> dict:
    """Return a Bedrock Converse API image content block."""
    data = path.read_bytes()
    fmt = "png" if path.suffix.lower() == ".png" else "jpeg"
    return {"image": {"format": fmt, "source": {"bytes": data}}}


def analyze_question(
    module_title: str,
    module_num: int,
    q_num: int,
    q_text: str,
    q_context: str,
    image_paths: list[Path],
    nav_hint: str,
    region: str = DEFAULT_REGION,
) -> dict:
    """
    Send one lab question + its screenshots to Kimi K2.5 on AWS Bedrock.

    Returns a dict with keys: images, explanation, answer_latex.
    Retries up to 3 times on invalid JSON or missing keys.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    user_prompt = f"""\
Analyze this forensic lab question and its associated screenshot(s).

## Question
Module {module_num} — {module_title}
Q{module_num}.{q_num}: {q_text}

## Answer Notes (from lab notebook — these are correct, use them)
{q_context.strip() if q_context.strip() else "(No explicit answer provided — read it from the screenshot.)"}

## AXIOM Navigation Steps (from the lab walkthrough)
{nav_hint}

## Provided Screenshots (listed in the order they appear above in the message)
{[p.name for p in image_paths] if image_paths else "(no screenshots for this question)"}

Produce the JSON output exactly as specified in the system prompt.
Carefully examine each screenshot to choose tight crop regions and precise
annotation rectangles that highlight the evidence answering the question.
"""

    # Build content: images first, then the text prompt
    content: list[dict] = []
    for img_path in image_paths:
        content.append(_image_block(img_path))
    content.append({"text": user_prompt})

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.converse(
                modelId=MODEL_ID,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"maxTokens": 2048, "temperature": 0.15},
            )
        except ClientError as exc:
            print(
                f"    [bedrock] Call failed (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            last_error = exc
            continue

        raw = "\n".join(
            block["text"]
            for block in response["output"]["message"].get("content", [])
            if "text" in block
        ).strip()

        # Strip markdown code fences if Kimi wraps the JSON
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)
        # If there's any text before the first { strip it
        brace = raw.find("{")
        if brace > 0:
            raw = raw[brace:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(
                f"    [json] Invalid JSON (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            print(f"    Raw (first 600 chars): {raw[:600]}", file=sys.stderr)
            last_error = exc
            continue

        required = {"images", "explanation", "answer_latex"}
        if required.issubset(result.keys()):
            return result

        print(
            f"    [json] Missing keys {required - result.keys()} "
            f"(attempt {attempt + 1}/3), retrying…",
            file=sys.stderr,
        )
        last_error = ValueError(f"missing keys: {required - result.keys()}")

    print(
        f"    [warn] All 3 attempts failed for Q{module_num}.{q_num}: {last_error}",
        file=sys.stderr,
    )
    return _fallback_analysis(image_paths, q_text, q_context)


def _fallback_analysis(
    image_paths: list[Path], q_text: str, q_context: str
) -> dict:
    """Minimal fallback when Kimi cannot be reached or returns invalid data."""
    images = []
    for p in image_paths:
        name = p.name
        output_name = (
            name.removeprefix("image-") if name.startswith("image-") else name
        )
        images.append(
            {
                "source": name,
                "crop": None,
                "annotations": [],
                "caption": "Axiom Examine v9.11: Screenshot for this question",
                "output_name": output_name,
            }
        )
    first_context_line = (
        q_context.strip().split("\n")[0] if q_context.strip() else ""
    )
    return {
        "images": images,
        "explanation": "",
        "answer_latex": first_context_line or q_text,
    }

"""
vision.py — Bedrock Converse image analysis using Qwen VL.

The pipeline sends a forensic lab question plus its screenshots to the
Bedrock Converse API and receives structured crop/annotation/caption/text
data as JSON.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "ap-south-1"
)
MODEL_ID = "qwen.qwen3-vl-235b-a22b"
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
COORDINATE SYSTEM
───────────────────────────────────────────────────────────────────────────

All x/y coordinates you return are FRACTIONS of the ORIGINAL image
dimensions (width W and height H), in the range [0.0, 1.0].

"crop" trims the image to a sub-region of the original.
"annotations" are drawn on the cropped image, but their coordinates are
still expressed in the ORIGINAL image coordinate space.

Crop loosely around the relevant region, usually leaving about 5 percent
margin, and exclude irrelevant UI where possible, especially the taskbar
and scrollbar. There should usually be some cropping if screenshots exist.

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
  "images": [
    {
      "source": "image-01.png",
      "crop": [left, top, right, bottom],
      "annotations": [[left, top, right, bottom]],
      "caption": "Axiom Examine v9.11: ...",
      "output_name": "01.png"
    }
  ],
  "explanation": "...",
  "answer_latex": "..."
}

## Field rules

"source"
  The original filename exactly as given to you.

"crop"
  [left, top, right, bottom] fractions of the original image. Use null only
  when the full image is truly needed.

"annotations"
  Array of [left, top, right, bottom] arrays in ORIGINAL image fractions.
  There must be at least one annotation if the question is answerable from
  the screenshot.

"caption"
  Must start with "Axiom Examine v9.11: ". Describe the artifact view and
  what is highlighted.

"output_name"
  Strip the "image-" prefix from the source filename:
  "image-01.png" → "01.png". Preserve unrelated prefixes.

"explanation"
  1 to 3 sentences in first-person plural. Describe the AXIOM navigation path
  and what the screenshot reveals.

"answer_latex"
  Content for the \\ansbox{} macro. Write a complete grammatical sentence.
  Wrap key answer values in \\ans{}. Use \\texttt{} for usernames, paths,
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
   be doubled. Write \\\\texttt{}, \\\\ans{}, \\\\newline,
   \\\\textbackslash{} — NOT \texttt{}, \ans{}, etc.
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


def analyze_question(
    module_title: str,
    module_num: int,
    q_num: int,
    q_text: str,
    q_context: str,
    image_paths: list[Path],
    nav_hint: str,
    region: str | None = None,
) -> dict:
    """
    Send one lab question plus its screenshots to Bedrock Converse.

    Returns a dict with keys: images, explanation, answer_latex.
    Retries up to 3 times on API failure, invalid JSON, or missing keys.
    """
    client = _get_client(region)
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
Carefully choose crop regions and annotation rectangles that highlight the
evidence answering the question.
"""

    content: list[dict] = [_image_block(path) for path in image_paths]
    content.append({"text": user_prompt})

    last_error: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(1.0)
        try:
            response = client.converse(
                modelId=MODEL_ID,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
            )
            time.sleep(REQUEST_DELAY)
        except (ClientError, BotoCoreError) as exc:
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

        try:
            result = _parse_json(raw)
        except json.JSONDecodeError as exc:
            print(
                f"    [json] Invalid JSON (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            print(f"    Raw (first 600 chars): {raw[:600]}", file=sys.stderr)
            last_error = exc
            continue

        required = {"images", "explanation", "answer_latex"}
        missing = required - result.keys()
        if not missing:
            return result

        print(
            f"    [json] Missing keys {missing} (attempt {attempt + 1}/3), retrying…",
            file=sys.stderr,
        )
        last_error = ValueError(f"missing keys: {missing}")

    raise RuntimeError(
        f"All 3 attempts failed for Q{module_num}.{q_num}: {last_error}"
    )

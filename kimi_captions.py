
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

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


SYSTEM_PROMPT = """\
You are a forensic lab report assistant. Your ONLY job is to analyze screenshots 
of various forensic tools and produce structured JSON that drives
an automated LaTeX report-generation pipeline.

Following is the context for what the questions the screenshots will be answering:
1. How to capture RAM Memory of a PC
2. How to add evidence items to FTK interface and view contents including deleted contents
3. How to create a physical Forensic copy of a device
4. How to create a hash set of selected files
5. How to create a logical image of a selected folder
6. Viewing file header of doc, pdf, jpg etc.
7. How to create RAM dump using Dumpit.
8. Compare the memory Footprint for FTKImager vs Dumpit. Show the memory footprint in MB document the procedure.


You must return a JSON object with the following structure:
{
    "likely_tools": [list of tools likely shown in the screenshot, e.g. "FTK Imager", "Dumpit", "Perfmon", "Excel"],
    "observations": [list of detailed observations about the screenshot, e.g. "There is a table showing file names, sizes, and hash values", "The title bar indicates..."],
    "question_answered": [the question from the above list that this screenshot most likely answers, e.g. "2. How to add evidence items to FTK interface..."],
    "summary": [a concise summary of the screenshot in 1-2 sentences]
}

Make sure to ONLY return the JSON object and nothing else. Do not include any explanations or commentary. The JSON should be properly formatted and parsable.
"""

def _parse_json(raw: str) -> dict:
    """Strip markdown fences, repair bare LaTeX backslashes, and parse JSON."""
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)
    # find the first JSON value opener (object or array)
    obj_start = raw.find("{")
    arr_start = raw.find("[")
    starts = [i for i in (obj_start, arr_start) if i >= 0]
    if starts:
        first = min(starts)
        if first > 0:
            raw = raw[first:]
    raw = raw.strip()
    def _unwrap(obj):
        if isinstance(obj, list):
            if len(obj) == 1:
                return obj[0]
            raise json.JSONDecodeError("expected object, got array with multiple elements", "", 0)
        return obj

    try:
        return _unwrap(json.loads(raw))
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\\(?=[a-zA-Z])', r'\\\\', raw)
        return _unwrap(json.loads(fixed))

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

def analyze_screenshot(
    client,
    image_paths: list[Path],
    nav_hint: str,
) -> dict:

    user_prompt = f"""\
Analyze this forensic lab screenshot.

## Navigation Steps (from the lab walkthrough)
{nav_hint}

## Provided Screenshots (listed in the same order as the image parts)
{[p.name for p in image_paths] if image_paths else "(no screenshots)"}

Produce the JSON output exactly as specified in the system prompt.
"""

    content: list[dict] = [_image_block(path) for path in image_paths]
    content.append({"text": user_prompt})

    last_error: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(1.0)
        try:
            print(f"Attempt {attempt + 1}/3: Calling Bedrock Converse API...", file=sys.stderr)
            raw = _call_converse(client, SYSTEM_PROMPT, content)
        except (ClientError, BotoCoreError) as exc:
            print(
                f"    [bedrock] call failed (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            last_error = exc
            continue

        try:
            result = _parse_json(raw)
        except json.JSONDecodeError as exc:
            print(
                f"    [json] invalid JSON (attempt {attempt + 1}/3): {exc}",
                file=sys.stderr,
            )
            print(f"    Raw: {raw}", file=sys.stderr)
            last_error = exc
            continue

        required = {"likely_tools", "observations", "question_answered", "summary"}
        missing = required - result.keys()
        if not missing:
            print(f"    [success]completed successfully", file=sys.stderr)
            return result

        print(
            f"    [json] missing keys {missing} (attempt {attempt + 1}/3), retrying…",
            file=sys.stderr,
        )
        last_error = ValueError(f"missing keys: {missing}")

    raise RuntimeError(
        f"Step 1 failed after 3 attempts: {last_error}"
    )

# read all parameters from command line and call analyze_screenshot, then save the result to a JSON file in the current directory with a timestamped name like "analysis_20240630_1530.json"
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze forensic lab screenshots with Bedrock Converse.")
    parser.add_argument("--region", type=str, default=DEFAULT_REGION, help="AWS region for Bedrock client")
    parser.add_argument("--nav_hint", type=str, default="", help="Path to file containing navigation hint from the lab walkthrough")
    parser.add_argument("images", nargs="*", type=Path, help="Paths to directory of screenshots to analyze")
    args = parser.parse_args()

    client = _get_client(args.region)
    with open(args.nav_hint, "r") as f:
        nav_hint = f.read()
    
    paths = []
    for img_path in args.images:
        if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            paths.append(img_path)
        elif img_path.is_dir():
            for file in sorted(img_path.iterdir()):
                if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    paths.append(file)
    if not paths:
        print("No valid image files found in the provided paths.", file=sys.stderr)
        sys.exit(1)

    #call the analyze_screenshot for each image and save the result to a JSON file
    for path in paths:
        result = analyze_screenshot(client, [path], nav_hint)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"analysis_{path.stem}_{timestamp}.json")
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Analysis for {path.name} saved to {output_path}")

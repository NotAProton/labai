"""
main.py — CLI entry point for the labber pipeline.

Usage (from the repo root):
    python -m labber.main [options]

Options:
    --lab-md PATH         Source markdown file  (default: context/lab5/lab5.md)
    --solve-md PATH       Navigation walkthrough (default: context/lab5/suggested_solve.md)
    --images-dir PATH     Directory with image-XX.png files  (default: context/lab5)
    --output-images PATH  Directory to write annotated images (default: work/lab5img)
    --output-tex PATH     Path for the generated .tex file   (default: work/main_generated.tex)
    --region STR          AWS Bedrock region (default: AWS_REGION or ap-south-1)
    --only STR            Process only listed questions, e.g. "3.1,3.2,4.1"
    --skip-vision         Skip Bedrock analysis; just copy+rename images (dry-run mode)
"""
import argparse
import shutil
import sys
from pathlib import Path

from .parser import parse_lab_md, extract_nav_hint
from .vision import analyze_question
from .annotator import process_image
from .latex_gen import generate_full_tex


def _parse_only(only_str: str | None) -> set[str] | None:
    """Parse --only "3.1,3.2,4.1" into a set {"3.1", "3.2", "4.1"}."""
    if not only_str:
        return None
    return {tok.strip() for tok in only_str.split(",") if tok.strip()}


def _dummy_analysis(question, img_paths: list[Path]) -> dict:
    """Minimal analysis used in --skip-vision mode or when vision fails."""
    images = []
    for p in img_paths:
        name = p.name
        output_name = (
            name.removeprefix("image-") if name.startswith("image-") else name
        )
        images.append(
            {
                "source": name,
                "crop": None,
                "annotations": [],
                "caption": f"Axiom Examine v9.11: Screenshot for question {question.number}",
                "output_name": output_name,
            }
        )
    first_line = (
        question.context.strip().split("\n")[0] if question.context.strip() else ""
    )
    return {
        "images": images,
        "explanation": "",
        "answer_latex": first_line or question.text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m labber.main",
        description="labber — AI forensics lab report pipeline (AWS Bedrock / Qwen VL)",
    )
    ap.add_argument("--lab-md",        default="context/lab7/lab7.md")
    ap.add_argument("--solve-md",      default="context/lab7/suggested_solve.md")
    ap.add_argument("--images-dir",    default="context/lab7")
    ap.add_argument("--output-images", default="work/lab7img")
    ap.add_argument("--output-tex",    default="work/main_generated.tex")
    ap.add_argument("--region",        default=None)
    ap.add_argument("--annotation-model", default=None, metavar="MODEL_ID",
                    help="Bedrock model for annotation stages (default: us.amazon.nova-lite-v1:0)")
    ap.add_argument("--annotation-region", default=None, metavar="REGION",
                    help="AWS region for annotation stages (default: us-east-1)")
    ap.add_argument(
        "--only",
        default=None,
        metavar="M.Q[,M.Q,…]",
        help='Only process these questions, e.g. "3.1,3.2,4.5"',
    )
    ap.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip Bedrock API calls; just copy/rename images (useful for testing)",
    )
    ap.add_argument(
        "--skip-manual-adjustment",
        action="store_true",
        help="Skip the manual bounding box GUI; run fully automated without pausing",
    )
    ap.add_argument(
        "--skip-modify-images",
        action="store_true",
        help=(
            "Skip all image modification (no crop, no Nova/bbox calls, no writes to "
            "output-images dir). Reads already-processed images from --output-images "
            "instead of --images-dir, sends them to Qwen for re-analysis/LaTeX, then "
            "compiles the report. The output-images folder is left untouched."
        ),
    )
    args = ap.parse_args()

    lab_md       = Path(args.lab_md)
    solve_md     = Path(args.solve_md)
    images_dir   = Path(args.images_dir)
    output_imgs  = Path(args.output_images)
    output_tex   = Path(args.output_tex)
    only_set     = _parse_only(args.only)

    # ── Validate inputs ──────────────────────────────────────
    for p, name in [(lab_md, "--lab-md"), (solve_md, "--solve-md"), (images_dir, "--images-dir")]:
        if not p.exists():
            sys.exit(f"[labber] ERROR: {name} path not found: {p}")

    # ── Load source data ─────────────────────────────────────
    print(f"[labber] Parsing {lab_md} …")
    modules = parse_lab_md(lab_md)
    print(f"[labber] Found {len(modules)} module(s), "
          f"{sum(len(m.questions) for m in modules)} question(s) total")

    print(f"[labber] Loading navigation hints from {solve_md} …")
    solve_text = solve_md.read_text(encoding="utf-8")

    output_imgs.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    # images_dir name for LaTeX \screenshot{} references
    images_dir_name = output_imgs.name  # e.g. "lab5img"

    # ── Process each module / question ───────────────────────
    modules_data: list[tuple[str, list[tuple[int, str, dict]]]] = []

    for module in modules:
        print(f"\n[labber] ═══ Module {module.module_number}: {module.title} ═══")
        questions_analyses: list[tuple[int, str, dict]] = []

        for question in module.questions:
            tag = f"{module.module_number}.{question.number}"

            if only_set and tag not in only_set:
                print(f"  [skip]  Q{tag}")
                continue

            short_text = question.text[:65] + ("…" if len(question.text) > 65 else "")
            print(f"  [Q{tag}]  {short_text}")

            # Resolve image paths
            img_paths: list[Path] = []
            if args.skip_modify_images:
                # Use already-processed images from output_imgs (strip "image-" prefix)
                for img_name in question.images:
                    stripped = img_name[len("image-"):] if img_name.startswith("image-") else img_name
                    p = output_imgs / stripped
                    if p.exists():
                        img_paths.append(p)
                    else:
                        print(f"    [warn] pre-annotated image not found: {p}", file=sys.stderr)
            else:
                for img_name in question.images:
                    p = images_dir / img_name
                    if p.exists():
                        img_paths.append(p)
                    else:
                        print(f"    [warn] image not found: {p}", file=sys.stderr)

            # Vision analysis
            if args.skip_vision or not img_paths:
                if not img_paths and not args.skip_vision:
                    print("    [info] no images — text-only question")
                analysis = _dummy_analysis(question, img_paths)
            else:
                nav_hint = extract_nav_hint(solve_text, question.number)
                region = args.region or "AWS default region"
                print(f"    [bedrock] sending {len(img_paths)} image(s) to Qwen VL in {region} …")
                try:
                    analysis = analyze_question(
                        module_title=module.title,
                        module_num=module.module_number,
                        q_num=question.number,
                        q_text=question.text,
                        q_context=question.context,
                        image_paths=img_paths,
                        nav_hint=nav_hint,
                        region=args.region,
                        annotation_region=args.annotation_region,
                        annotation_model=args.annotation_model,
                        allow_manual_adjustment=not args.skip_manual_adjustment,
                        skip_modify_images=args.skip_modify_images,
                    )
                    n_imgs = len(analysis.get("images") or [])
                    print(f"    [bedrock] ✓ received analysis for {n_imgs} image(s)")
                except Exception as exc:
                    print(f"    [error] vision failed: {exc}", file=sys.stderr)
                    analysis = _dummy_analysis(question, img_paths)

            # Process images (crop + annotate → output_imgs/)
            if args.skip_modify_images:
                # Images are already in output_imgs — nothing to write
                n_imgs = len(analysis.get("images") or [])
                if n_imgs:
                    print(f"    [skip-modify-images] {n_imgs} image(s) left unchanged in {output_imgs}")
            else:
                for img_item in analysis.get("images") or []:
                    output_name = (img_item.get("output_name") or "").strip()
                    src_name    = (img_item.get("source") or "").strip()
                    if not output_name or not src_name:
                        continue

                    src_path = images_dir / src_name
                    dst_path = output_imgs / output_name

                    if not src_path.exists():
                        print(f"    [warn] source image missing: {src_path}", file=sys.stderr)
                        continue

                    if args.skip_vision:
                        shutil.copy2(src_path, dst_path)
                        print(f"    [copy]  {src_name} → {output_name}")
                    else:
                        try:
                            process_image(src_path, img_item, dst_path)
                            cropped = img_item.get("crop") is not None
                            annotated = bool(img_item.get("annotations"))
                            flags = []
                            if cropped:
                                flags.append("cropped")
                            if annotated:
                                flags.append(f"{len(img_item['annotations'])} annotation(s)")
                            flag_str = ", ".join(flags) if flags else "full image"
                            print(f"    [img]   {src_name} → {output_name} ({flag_str})")
                        except Exception as exc:
                            print(f"    [warn] image processing failed for {src_name}: {exc}",
                                  file=sys.stderr)
                            shutil.copy2(src_path, dst_path)
                            print(f"    [copy]  {src_name} → {output_name} (fallback)")

            questions_analyses.append((question.number, analysis["question_summary"], analysis))

        if questions_analyses:
            modules_data.append((module.title, questions_analyses))

    if not modules_data:
        sys.exit("[labber] Nothing to process. Check --only filter or lab-md content.")

    # ── Generate LaTeX ────────────────────────────────────────
    print(f"\n[labber] Generating LaTeX … ({images_dir_name}/)")
    tex = generate_full_tex(modules_data, images_dir=images_dir_name)
    output_tex.write_text(tex, encoding="utf-8")
    print(f"[labber] Written → {output_tex}")
    print(f"\n[labber] ✓ Done!  To compile:")
    print(f"           cd {output_tex.parent} && pdflatex {output_tex.name}")


if __name__ == "__main__":
    main()

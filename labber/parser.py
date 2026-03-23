"""
parser.py — Parse lab5.md and suggested_solve.md into structured data.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    number: int
    text: str         # Raw question text from the numbered line
    context: str      # Answer/notes lines mixed in (non-image text)
    images: list[str] = field(default_factory=list)  # e.g. ["image-01.png"]


@dataclass
class Module:
    title: str            # e.g. "OS Part 1"
    module_number: int    # e.g. 3
    questions: list[Question] = field(default_factory=list)


def parse_lab_md(path: str | Path) -> list[Module]:
    """
    Parse a lab markdown file (lab5.md style) into Module/Question objects.

    Module headers:  ### **Module N – Title**
    Questions:       N.\\t Question text  (tab or 2+ spaces after the number+dot)
    Images:          ![alt](filename.png)
    Context:         any other non-blank line after a question start
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    modules: list[Module] = []
    current_module: Module | None = None
    current_q: Question | None = None
    module_counter = 2  # \setcounter{modnum}{2} → first \module{} = Module 3

    def flush_q():
        nonlocal current_q
        if current_q is not None and current_module is not None:
            current_module.questions.append(current_q)
            current_q = None

    for line in lines:
        # Module header: ### **Module N – Title** or ### **Module N: Title**
        mod_m = re.match(r"^###\s+\*\*Module\s+\d+\s*[–\-:]\s*(.+?)\*\*\s*$", line)
        if mod_m:
            flush_q()
            module_counter += 1
            current_module = Module(
                title=mod_m.group(1).strip(),
                module_number=module_counter,
            )
            modules.append(current_module)
            continue

        if current_module is None:
            continue

        # Question line: "N.  text" or "N.\ttext" (tab or 2+ spaces)
        q_m = re.match(r"^(\d+)\.\s{1,}(.+)$", line)
        if q_m:
            flush_q()
            current_q = Question(
                number=int(q_m.group(1)),
                text=q_m.group(2).strip(),
                context="",
                images=[],
            )
            continue

        if current_q is None:
            continue

        # Image reference line
        img_refs = re.findall(r"!\[.*?\]\(([^)]+\.png)\)", line)
        if img_refs:
            current_q.images.extend(img_refs)
            continue

        # Everything else is answer/context
        stripped = line.strip()
        if stripped:
            current_q.context += stripped + "\n"

    flush_q()
    return modules


def extract_nav_hint(solve_text: str, q_num: int) -> str:
    """
    Extract the navigation steps for a specific question number from
    the suggested_solve.md text.

    Looks for headers like:
        **1. Title**          (single question)
        **8, 9, 10, 11. ..** (range of questions)
    """
    lines = solve_text.splitlines()
    result: list[str] = []
    in_section = False

    for line in lines:
        header_m = re.match(r"^\*\*(\d+(?:,\s*\d+)*)\.", line)
        if header_m:
            nums = [int(n.strip()) for n in header_m.group(1).split(",")]
            if q_num in nums:
                in_section = True
                result = [line]
                continue
            elif in_section:
                break  # hit the next section
        if in_section:
            result.append(line)

    if result:
        # Trim trailing blank lines
        while result and not result[-1].strip():
            result.pop()
        return "\n".join(result)

    return "(No specific navigation hint found — refer to the suggested solve document.)"

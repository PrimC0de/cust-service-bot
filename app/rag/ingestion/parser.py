"""Parse knowledge files into titled, hierarchical sections."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.models import ParsedDocument, ParsedSection


ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^(?:[-*+]\s|\d+[.)]\s)")


def load_taxonomy(path: Path) -> set[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        (category["slug"], sub_intent["slug"])
        for category in raw.get("categories", [])
        for sub_intent in category.get("sub_intents", [])
    }


def _plain_heading(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.endswith(":")
        and len(stripped) <= 80
        and not LIST_ITEM.match(stripped)
    )


def parse_document(path: Path, knowledge_dir: Path) -> ParsedDocument:
    lines = path.read_text(encoding="utf-8").splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        raise ValueError(f"Empty knowledge document: {path}")

    first_heading = ATX_HEADING.match(lines[first].strip())
    title = first_heading.group(2).strip() if first_heading else lines[first].strip()
    stack: list[tuple[int, str]] = []
    section_path: tuple[str, ...] = ("Overview",)
    content: list[str] = []
    sections: list[ParsedSection] = []

    def flush() -> None:
        text = "\n".join(content).strip()
        if text:
            sections.append(ParsedSection(path=section_path, text=text))
        content.clear()

    for line in lines[first + 1 :]:
        stripped = line.strip()
        heading = ATX_HEADING.match(stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            name = heading.group(2).strip()
            stack[:] = [(old_level, old_name) for old_level, old_name in stack if old_level < level]
            stack.append((level, name))
            section_path = tuple(name for _, name in stack)
        elif _plain_heading(stripped):
            flush()
            name = stripped[:-1].strip()
            stack[:] = [(2, name)]
            section_path = (name,)
        else:
            content.append(line.rstrip())
    flush()

    if not sections:
        raise ValueError(f"No section content found: {path}")
    return ParsedDocument(
        title=title,
        category=path.parent.name,
        sub_category=path.stem,
        source=str(path.relative_to(knowledge_dir.parent.parent.parent)),
        sections=tuple(sections),
    )


def parse_documents(
    knowledge_dir: Path,
    taxonomy_path: Path,
) -> list[ParsedDocument]:
    valid = load_taxonomy(taxonomy_path)
    documents: list[ParsedDocument] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        key = (path.parent.name, path.stem)
        if key not in valid:
            raise ValueError(f"Knowledge document is absent from taxonomy: {path}")
        documents.append(parse_document(path, knowledge_dir))
    if not documents:
        raise ValueError(f"No knowledge documents found in {knowledge_dir}")
    return documents


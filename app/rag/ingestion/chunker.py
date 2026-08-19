"""Structure-aware knowledge chunking."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import KnowledgeChunk, ParsedDocument, ParsedSection


CHUNKING_METHOD = "atomic-intent-utterances-v1"


def _related(left: ParsedSection, right: ParsedSection) -> bool:
    return left.path[:-1] == right.path[:-1]


def _merged_path(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    parent = left[:-1]
    return parent + (f"{left[-1]} / {right[-1]}",)


def merge_small_sections(
    sections: tuple[ParsedSection, ...],
    chunk_size: int,
) -> list[ParsedSection]:
    merged: list[ParsedSection] = []
    for section in sections:
        if merged and _related(merged[-1], section):
            combined = f"{merged[-1].text.rstrip()}\n\n{section.text.lstrip()}"
            if len(combined) <= chunk_size:
                merged[-1] = ParsedSection(
                    path=_merged_path(merged[-1].path, section.path),
                    text=combined,
                )
                continue
        merged.append(section)
    return merged


def chunk_documents(
    documents: list[ParsedDocument],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[KnowledgeChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for section in merge_small_sections(document.sections, chunk_size):
            pieces = (
                [section.text]
                if len(section.text) <= chunk_size
                else splitter.split_text(section.text)
            )
            for piece in pieces:
                text = piece.strip()
                if not text:
                    continue
                path_text = " > ".join(section.path)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=len(chunks),
                        text=text,
                        embedding_text=f"{document.title}\n{path_text}\n{text}",
                        category=document.category,
                        sub_category=document.sub_category,
                        source=document.source,
                        document_title=document.title,
                        section_path=section.path,
                    )
                )
    return [replace(chunk, chunk_id=index) for index, chunk in enumerate(chunks)]


def chunk_intent_documents(documents: list[ParsedDocument]) -> list[KnowledgeChunk]:
    """Keep each intent playbook as one retrieval unit."""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        text = "\n\n".join(
            f"{' > '.join(section.path)}\n{section.text}"
            for section in document.sections
        )
        retrieval_lines: list[str] = []
        for section in document.sections:
            for line in section.text.splitlines():
                if line.startswith("Classification cues:"):
                    retrieval_lines.extend(
                        phrase.strip()
                        for phrase in line.split(":", 1)[1].split(";")
                    )
                elif section.path[0] == "Customer Utterance Examples" and line.startswith("- "):
                    retrieval_lines.append(line[2:].split(" —", 1)[0])
        retrieval_text = "\n".join(dict.fromkeys(retrieval_lines))
        chunks.append(
            KnowledgeChunk(
                chunk_id=len(chunks),
                text=text,
                embedding_text=retrieval_text,
                category=document.category,
                sub_category=document.sub_category,
                source=document.source,
                document_title=document.title,
                section_path=("Intent Playbook",),
            )
        )
    return chunks


def _pipe_records(path: Path, prefix: str, fields: int) -> list[list[str]]:
    records = [
        [value.strip() for value in line.split(" | ")]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if any(len(record) != fields for record in records):
        raise ValueError(f"Invalid record in {path}")
    return records


def load_atomic_intent_chunks(knowledge_dir: Path) -> list[KnowledgeChunk]:
    """Load the layered intent KB without embedding catalogs or behavior rules."""
    catalog_path = knowledge_dir / "intent-catalog.txt"
    examples_path = knowledge_dir / "atomic-utterance-examples.txt"
    rules_path = knowledge_dir / "behavior-rules.txt"
    ambiguity_path = knowledge_dir / "ambiguity-map.txt"
    for path in (catalog_path, examples_path, rules_path, ambiguity_path):
        if not path.is_file():
            raise ValueError(f"Missing intent knowledge layer: {path}")

    catalog: dict[str, list[str]] = {}
    intent_id: str | None = None
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## INT_"):
            intent_id = line[3:].strip()
            catalog[intent_id] = [line]
        elif intent_id is not None:
            catalog[intent_id].append(line)

    rules: dict[str, list[str]] = {}
    for record in _pipe_records(rules_path, "RULE_", 9):
        rules.setdefault(record[1], []).append(" | ".join(record))

    ambiguities: dict[str, list[str]] = {}
    for record in _pipe_records(ambiguity_path, "AMB_", 8):
        line = " | ".join(record)
        ambiguities.setdefault(record[1], []).append(line)
        ambiguities.setdefault(record[2], []).append(line)

    chunks: list[KnowledgeChunk] = []
    seen: set[tuple[str, str]] = set()
    for record in _pipe_records(examples_path, "EX_", 9):
        _, utterance, record_intent_id, intent_name, _, example_type, source, _, _ = record
        key = (record_intent_id, utterance.casefold())
        if not utterance or key in seen:
            raise ValueError(f"Empty or duplicate atomic utterance: {record}")
        if record_intent_id not in catalog or record_intent_id not in rules:
            raise ValueError(f"Missing catalog or rules for {record_intent_id}")
        source_path = knowledge_dir / source
        if not source_path.is_file():
            raise ValueError(f"Missing original source: {source_path}")
        seen.add(key)
        context = (
            "Intent Catalog\n"
            + "\n".join(catalog[record_intent_id]).strip()
            + "\n\nBehavior Rules\n"
            + "\n".join(rules[record_intent_id])
            + "\n\nAmbiguity Map\n"
            + "\n".join(ambiguities.get(record_intent_id, ["(none)"]))
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=len(chunks),
                text=context,
                embedding_text=utterance,
                category="intent-knowledge",
                sub_category=Path(source).stem,
                source=f"data/raw/intent-knowledge/{source}",
                document_title=intent_name,
                section_path=("Atomic Utterance", example_type),
            )
        )
    if not chunks:
        raise ValueError(f"No atomic utterance examples found in {examples_path}")
    return chunks

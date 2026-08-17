"""Structure-aware knowledge chunking."""

from __future__ import annotations

from dataclasses import replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import KnowledgeChunk, ParsedDocument, ParsedSection


CHUNKING_METHOD = "structure-aware-v1"


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


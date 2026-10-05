from __future__ import annotations

from pathlib import Path

from rag_agent.domain.models import Document


class TextFileLoader:
    suffixes = {".txt", ".md", ".markdown"}

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    def load(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, source=path.name)] if text.strip() else []


class PdfLoader:
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path) -> list[Document]:
        from pypdf import PdfReader

        docs: list[Document] = []
        for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(text=text, source=path.name, metadata={"page": page_no}))
        return docs

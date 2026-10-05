from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvalSample(BaseModel):
    question: str
    reference: str
    source: str | None = None  # file the answer lives in (enables hit-rate / MRR)


def load_test_set(path: Path) -> list[EvalSample]:
    return [EvalSample(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]

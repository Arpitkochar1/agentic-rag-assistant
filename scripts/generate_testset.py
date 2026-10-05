"""Synthesise a Q/A test set from your own chunks so you can reach e.g. 30 questions fast.
ALWAYS spot-check the output - synthetic references can be wrong.

python scripts/generate_testset.py --n 30 --out evaluation/test_set.json
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.container import Container  # noqa: E402
from rag_agent.domain.models import ChatMessage  # noqa: E402

PROMPT = (
    "From the passage, write ONE specific question a user could ask that is answerable ONLY from this passage, "
    "plus a short, correct reference answer. Return strict JSON: {\"question\": \"...\", \"reference\": \"...\"}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("evaluation/test_set.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    c = Container()
    c.ensure_index()
    chunks = list(c.keyword_index._chunks.values())  # noqa: SLF001 (script-only convenience)
    random.Random(args.seed).shuffle(chunks)
    samples = []
    for chunk in chunks:
        if len(samples) >= args.n:
            break
        if len(chunk.text) < 200:
            continue
        out = c.llm.invoke([ChatMessage(role="system", content=PROMPT), ChatMessage(role="user", content=chunk.text)])
        m = re.search(r"\{.*\}", out, re.S)
        if not m:
            continue
        try:
            d = json.loads(m.group())
            samples.append({"question": d["question"], "reference": d["reference"], "source": chunk.source})
        except (json.JSONDecodeError, KeyError):
            continue
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()

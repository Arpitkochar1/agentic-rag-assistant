"""Compare retrieval strategies with retrieval metrics + RAGAS.

pip install -r requirements-extras.txt
python scripts/evaluate.py --test-set evaluation/test_set.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.container import Container  # noqa: E402
from rag_agent.evaluation.dataset import load_test_set  # noqa: E402
from rag_agent.evaluation.ragas_eval import RagasEvaluator  # noqa: E402
from rag_agent.evaluation.retrieval_metrics import hit_rate_and_mrr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", type=Path, default=Path("evaluation/test_set.json"))
    ap.add_argument("--strategies", nargs="+", default=["vector", "hybrid", "hybrid_rerank"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ragas", action="store_true", help="retrieval metrics only (free, no LLM calls)")
    args = ap.parse_args()

    c = Container()
    c.ensure_index()
    samples = load_test_set(args.test_set)[: args.limit]
    print(f"{len(samples)} questions, index={c.vector_store.count()} chunks\n")

    results: dict[str, dict[str, float]] = {}
    for strategy in args.strategies:
        retriever = c.retriever(strategy)
        metrics = hit_rate_and_mrr(retriever, samples, args.k)
        if not args.no_ragas:
            rows = []
            for s in samples:
                ctxs = retriever.retrieve(s.question, args.k)
                answer = c.generator.generate(s.question, ctxs)
                rows.append(
                    {
                        "user_input": s.question,
                        "response": answer,
                        "retrieved_contexts": [x.chunk.text for x in ctxs],
                        "reference": s.reference,
                    }
                )
            metrics.update(RagasEvaluator(c.judge_llm, c.embedder).evaluate(rows))
        results[strategy] = metrics
        print(f"[{strategy}] {metrics}")

    # ---- report ----
    cols = sorted({k for m in results.values() for k in m})
    print("\n| strategy | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for strat, m in results.items():
        print(f"| {strat} | " + " | ".join(f"{m.get(c_, float('nan')):.3f}" for c_ in cols) + " |")

    base = results.get("vector")
    if base:
        print("\nRelative improvement over vector baseline:")
        for strat, m in results.items():
            if strat == "vector":
                continue
            deltas = {k: f"{(m[k] - base[k]) / base[k] * 100:+.1f}%" for k in m if k in base and base[k]}
            print(f"  {strat}: {deltas}")

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run_{int(time.time())}.json"
    out.write_text(json.dumps({"n": len(samples), "k": args.k, "results": results}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

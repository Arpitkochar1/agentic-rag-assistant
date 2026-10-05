"""python scripts/ingest.py [--reset] [PATH ...]   (default: data/documents)"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_agent.container import Container  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--reset", action="store_true", help="wipe the index first")
    args = ap.parse_args()

    c = Container()
    if args.reset:
        c.pipeline.reset()
    paths = args.paths or [c.settings.data_dir]
    total = None
    for p in paths:
        report = c.pipeline.ingest_directory(p) if p.is_dir() else c.pipeline.ingest_paths([p])
        print(f"{p}: {report.files} files, {report.chunks_added} new chunks, skipped={report.skipped}")
    print(f"Index now holds {c.vector_store.count()} chunks.")


if __name__ == "__main__":
    main()

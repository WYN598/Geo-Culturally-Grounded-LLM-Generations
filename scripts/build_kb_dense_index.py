import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import DenseKBIndex, OpenAIEmbedder


def run(args: argparse.Namespace) -> None:
    embedder = OpenAIEmbedder(
        model=args.embedding_model,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
    )
    index = DenseKBIndex.from_jsonl(
        kb_path=args.kb_path,
        embedder=embedder,
        use_faiss=bool(args.use_faiss),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    index.save(args.out_dir)

    summary = {
        "kb_path": args.kb_path,
        "out_dir": args.out_dir,
        "num_docs": len(index.docs),
        "dim": index.dim,
        "embedding_model": embedder.model,
        "faiss_enabled": bool(index.use_faiss),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dense vector index for KB-grounding.")
    parser.add_argument("--kb-path", default="data/kb_full.jsonl")
    parser.add_argument("--out-dir", default="data/kb_dense_index")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dimensions", type=int, default=0)
    parser.add_argument("--use-faiss", action="store_true")
    args = parser.parse_args()
    run(args)

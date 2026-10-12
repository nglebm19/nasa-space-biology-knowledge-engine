"""Search the FAISS index for chunks relevant to a question.

Usage:
    python search.py "How does microgravity affect skeletal muscle?"
    python search.py "..." -k 10
    python search.py --eval            # run every query in eval_queries.txt
"""

import argparse
import json
import textwrap
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
INDEX_FILE = Path("data/index.faiss")
EVAL_FILE = Path("eval_queries.txt")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match build_index.py


def load():
    if not INDEX_FILE.exists():
        raise SystemExit(f"{INDEX_FILE} not found. Run build_index.py first.")
    index = faiss.read_index(str(INDEX_FILE))
    chunks = [json.loads(line) for line in CHUNKS_FILE.open()]
    return index, chunks, SentenceTransformer(MODEL_NAME)


def search(question, index, chunks, model, k):
    vec = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(vec, k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]


def print_results(question, results):
    print(f"\nQ: {question}\n")
    for rank, (score, c) in enumerate(results, 1):
        heading = f" / {c['heading']}" if c.get("heading") else ""
        print(f"{rank}. score={score:.3f}  [{c['pmcid']} / {c['section']}{heading}]")
        print(f"   {c['title']}")
        print(f"   {c['url']}")
        print(textwrap.indent(textwrap.fill(c["text"][:400] + " ...", 100), "   > "))
        print()


def read_eval_queries():
    """Each line: question | expected PMC IDs (comma separated)."""
    queries = []
    for line in EVAL_FILE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        question, _, expected = line.partition("|")
        queries.append((question.strip(), {e.strip() for e in expected.split(",") if e.strip()}))
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("-k", type=int, default=5, help="number of chunks to return (default 5)")
    parser.add_argument("--eval", action="store_true", help="run the queries in eval_queries.txt")
    args = parser.parse_args()
    if not args.question and not args.eval:
        parser.error("give a question or --eval")

    index, chunks, model = load()

    if args.question:
        print_results(args.question, search(args.question, index, chunks, model, args.k))
        return

    hits = 0
    for question, expected in read_eval_queries():
        results = search(question, index, chunks, model, args.k)
        print_results(question, results)
        found = expected & {c["pmcid"] for _, c in results}
        hits += bool(found)
        print(f"   expected {sorted(expected)} -> {'HIT ' + str(sorted(found)) if found else 'MISS'}")
        print("-" * 100)
    print(f"\n{hits}/{len(read_eval_queries())} queries had an expected paper in the top {args.k}")


if __name__ == "__main__":
    main()

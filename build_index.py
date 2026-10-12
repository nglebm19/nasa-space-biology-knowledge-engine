"""Clean and chunk the downloaded papers, embed the chunks and build a FAISS index.

Reads data/raw/<PMCID>.json for the IDs in papers.txt, writes data/chunks.jsonl and data/index.faiss.
Usage: python build_index.py
"""

import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from fetch_papers import PAPERS_FILE, read_pmcids

RAW_DIR = Path("data/raw")
CHUNKS_FILE = Path("data/chunks.jsonl")
INDEX_FILE = Path("data/index.faiss")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# BioC section types worth searching; references, acknowledgments, author info etc. are dropped.
KEEP_SECTIONS = {"ABSTRACT", "INTRO", "METHODS", "RESULTS", "DISCUSS", "CONCL", "CASE", "FIG"}
KEEP_TYPES = {"abstract", "paragraph", "fig_caption"}

CHUNK_WORDS = 250
OVERLAP_WORDS = 50


def clean(text):
    text = re.sub(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]", "", text)  # numeric citations like [3], [4-6]
    return re.sub(r"\s+", " ", text).strip()


def load_paper(path):
    doc = json.loads(path.read_text())[0]["documents"][0]
    pmcid = doc["id"] if doc["id"].startswith("PMC") else f"PMC{doc['id']}"
    title, sections = "", []  # sections: list of (section_type, [paragraph, ...])

    for p in doc["passages"]:
        sec = p["infons"].get("section_type", "")
        kind = p["infons"].get("type", "")
        if sec == "TITLE" and kind == "front":
            title = clean(p["text"])
        if sec not in KEEP_SECTIONS or kind not in KEEP_TYPES:
            continue
        text = clean(p.get("text", ""))
        if not text:
            continue
        if sections and sections[-1][0] == sec:
            sections[-1][1].append(text)
        else:
            sections.append((sec, [text]))

    return pmcid, title, sections


def chunk_words(words):
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        yield words[start:start + CHUNK_WORDS]
        if start + CHUNK_WORDS >= len(words):
            break


def make_chunks(pmcid, title, sections):
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    for sec, paragraphs in sections:
        words = " ".join(paragraphs).split()
        for piece in chunk_words(words):
            yield {"pmcid": pmcid, "title": title, "url": url, "section": sec, "text": " ".join(piece)}


def main():
    chunks = []
    for pmcid in read_pmcids(PAPERS_FILE):
        path = RAW_DIR / f"{pmcid}.json"
        if not path.exists():
            print(f"{pmcid}: missing, run fetch_papers.py (skipped)")
            continue
        pmcid, title, sections = load_paper(path)
        paper_chunks = list(make_chunks(pmcid, title, sections))
        print(f"{pmcid}: {len(paper_chunks):4d} chunks  {title[:70]}")
        chunks.extend(paper_chunks)

    if not chunks:
        raise SystemExit(f"No papers in {RAW_DIR}. Run fetch_papers.py first.")

    print(f"\nEmbedding {len(chunks)} chunks with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # inner product == cosine similarity
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))

    # Line i of chunks.jsonl is vector i of the index.
    with CHUNKS_FILE.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {index.ntotal} vectors to {INDEX_FILE} and chunks to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()

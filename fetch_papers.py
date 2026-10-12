"""Download full text for the PMC IDs in papers.txt as BioC JSON.

Files are cached in data/raw/<PMCID>.json; existing files are skipped.
Usage: python fetch_papers.py
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PAPERS_FILE = Path("papers.txt")
RAW_DIR = Path("data/raw")
BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"


def read_pmcids(path):
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def fetch(pmcid):
    with urllib.request.urlopen(BIOC_URL.format(pmcid=pmcid), timeout=30) as resp:
        body = resp.read().decode("utf-8")
    # The API answers 200 with a plain-text message when an article is not available.
    if body.lstrip().startswith("[Error]"):
        raise ValueError("not in the PMC open-access text set; pick another paper")
    data = json.loads(body)
    if not data or not data[0].get("documents"):
        raise ValueError("no documents in response")
    return data


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pmcids = read_pmcids(PAPERS_FILE)
    failed = []

    for pmcid in pmcids:
        out = RAW_DIR / f"{pmcid}.json"
        if out.exists():
            print(f"cached   {pmcid}")
            continue
        try:
            data = fetch(pmcid)
        except (urllib.error.URLError, ValueError) as e:
            print(f"FAILED   {pmcid}: {e}")
            failed.append(pmcid)
            continue
        out.write_text(json.dumps(data))
        print(f"fetched  {pmcid}")
        time.sleep(0.4)  # stay under NCBI's 3 requests/second limit

    print(f"\n{len(pmcids) - len(failed)}/{len(pmcids)} papers available in {RAW_DIR}")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()

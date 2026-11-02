"""Download the public raw datasets into data/raw/ (skips files that already exist).

    python -m src.download_data

* COMI-LINGUA (LingoIITGN/COMI-LINGUA, CC-BY-4.0): LID and TN subsets, Hinglish.
* Tanglish-Corpus-185k (vishnu-n/Tanglish-Corpus-185k, CC-BY-4.0): Romanized Tamil-English sentences.
"""
from __future__ import annotations

import urllib.request

from .config import RAW

FILES = {
    "comi_lingua/LID_train.csv": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/LID_train.csv",
    "comi_lingua/LID_test.csv": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/LID_test.csv",
    "comi_lingua/TN_train.csv": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/TN_train.csv",
    "comi_lingua/TN_test.csv": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/TN_test.csv",
    "comi_lingua/README.md": "https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA/resolve/main/README.md",
    "tanglish/tanglish_corpus.jsonl": "https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k/resolve/main/tanglish_corpus.jsonl",
}


def main() -> None:
    for rel, url in FILES.items():
        dest = RAW / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch {rel} ...", flush=True)
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    main()

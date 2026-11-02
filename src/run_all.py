"""Reproduce the whole project: data -> language ID -> normalization -> language models.

    python -m src.run_all                 # everything (about 40-60 min on an Apple-silicon laptop)
    python -m src.run_all --skip-data     # reuse data/ and only retrain
    python -m src.run_all --only lm       # one stage: data | lid | norm | lm
"""
from __future__ import annotations

import argparse
import time

from . import data_prep, download_data, train_lid, train_lm, train_normalizer
from .config import WORD_LABELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--only", choices=["data", "lid", "norm", "lm"])
    ap.add_argument("--lid-epochs", type=int, default=10)
    ap.add_argument("--lm-epochs", type=int, help="override per-variant LSTM epochs")
    ap.add_argument("--lm-time-budget-min", type=float, default=45)
    a = ap.parse_args()
    want = lambda s: a.only in (None, s)          # noqa: E731

    t0 = time.time()
    if want("data") and not a.skip_data:
        download_data.main()
        data_prep.main()
    if want("lid"):
        for v in WORD_LABELS:
            train_lid.run(v, a.lid_epochs)
    if want("norm"):
        train_normalizer.main()
    if want("lm"):
        for v in WORD_LABELS:
            train_lm.run(v, a.lm_epochs, a.lm_time_budget_min)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> see results/*.json, then `streamlit run app.py`")


if __name__ == "__main__":
    main()

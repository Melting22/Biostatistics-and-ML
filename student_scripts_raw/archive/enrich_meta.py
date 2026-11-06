# -*- coding: utf-8 -*-
"""Enrich tensor rows with clinical metadata by subject id.

Reads measurements_with_legend.npz + nosologies.json READ-ONLY.
Writes new artifact only: results/meta_enrichment.csv (+ a JSON report).
Does not modify main.py or the source master_data.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from config import SRC_JSON, SRC_NPZ, PLOTS_DIR

REPO = Path(__file__).resolve().parent
OUT = REPO / "results"


def lookup(rec, *names):
    """Find a string value by key name, searching top-level scalars
    and one level into child dict blocks (case-insensitive)."""
    if not isinstance(rec, dict):
        return None
    wanted = [n.lower() for n in names]
    # 1) top-level scalars
    low = {str(k).lower(): v for k, v in rec.items()}
    for n in wanted:
        v = low.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 2) inside child dict blocks
    for v in rec.values():
        if isinstance(v, dict):
            sub = {str(kk).lower(): vv for kk, vv in v.items()}
            for n in wanted:
                w = sub.get(n)
                if isinstance(w, str) and w.strip():
                    return w.strip()
    return None


def build():
    recs = json.load(open(SRC_JSON, encoding="utf-8"))
    rows = list(np.load(SRC_NPZ)["rows"])
    rows = [str(r) for r in rows]

    data = []
    for sid in rows:
        rec = recs.get(sid, {})
        scanner_type = lookup(rec, "dataset")                      # TCP / UCLA
        site = lookup(rec, "site")                                 # 1 / 2 (TCP)
        serial = lookup(rec, "scannerserialnumber")                # UCLA scanner serial 35343/35426
        dx = lookup(rec, "primary_dx", "diagnosis")                # raw diagnosis text
        sex = lookup(rec, "gender", "sex")                         # M / F
        age = lookup(rec, "age")                                   # years (string)
        # Unified REAL batch key: no synthetic values, no imputation.
        # TCP cohort identifies its scanner by center Site (1/2);
        # UCLA cohort identifies its scanner by ScannerSerialNumber (35343/35426).
        if scanner_type == "TCP" and site:
            batch = f"TCP_site{site}"
        elif scanner_type == "UCLA" and serial:
            batch = f"UCLA_scn{serial}"
        else:
            batch = None
        data.append({
            "subject_id": sid,
            "scanner_type": scanner_type,
            "batch": batch,               # complete, real, variability-preserving
            "site_batch": site,           # TCP only
            "scanner_serial": serial,     # UCLA only
            "primary_dx": dx,             # raw text (999 kept as-is)
            "gender": sex,                # raw M/F/O
            "age_years": age,
        })

    df = pd.DataFrame(data)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "meta_enrichment.csv", index=False)

    # completeness report (no imputation yet)
    report = {c: int(df[c].replace("", np.nan).isna().sum()) for c in df.columns if c != "subject_id"}
    print("rows:", len(df))
    print("nulls per column:", report)
    print("\nscanner_type:", dict(Counter(df["scanner_type"].fillna("NA"))))
    print("batch       :", dict(Counter(df["batch"].fillna("NA"))))
    print("site_batch  :", dict(Counter(df["site_batch"].fillna("NA"))))
    print("scanner_serial present:", int(df["scanner_serial"].notna().sum()))
    print("primary_dx distinct:", df["primary_dx"].nunique())
    print("gender      :", dict(Counter(df["gender"].fillna("NA"))))
    print("\nhead:")
    print(df.head(12).to_string(index=False))
    return df


def enrich_dataframe(raw_data):
    """Attach the meta columns to YOUR DataFrame (index == tensor rows).

    Returns a NEW enriched copy; the input frame is not mutated and no
    source file is touched."""
    meta = build().set_index("subject_id")
    cols = ["scanner_type", "batch", "site_batch", "scanner_serial",
            "primary_dx", "gender", "age_years"]
    out = raw_data.copy()
    idx = out.index.astype(str)
    for c in cols:
        out[c] = idx.map(meta[c])
    return out


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatic dataset builder from protein names for synthetic biology SNR/performance prediction.

Starting point:
- Only protein names, e.g. SrpR, PhlF, AmeR, BetI, QacR, AmtR, LmrA.

Main idea:
- Use proteins as initial biological components.
- Retrieve sequence and metadata from UniProt.
- Compute physicochemical descriptors from protein sequences.
- Search Cello/UCF JSON files for experimentally characterized gate parameters.
- Use existing metrics as targets:
    1. SNR, if explicitly available;
    2. on/off ratio, if ymax and ymin are available;
    3. dynamic range, if ymax and ymin are available;
    4. any numeric performance-like field discovered in UCF.

No SBML simulation is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from Bio.SeqUtils.ProtParam import ProteinAnalysis


# ============================================================
# User default proteins
# ============================================================

DEFAULT_PROTEINS = [
    "SrpR",
    "PhlF",
    "AmeR",
    "BetI",
    "QacR",
    "AmtR",
    "LmrA",
]


# ============================================================
# Basic helpers
# ============================================================

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def sanitize_feature_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def clean_protein_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return "".join([aa for aa in seq if aa in VALID_AA])


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def flatten_json(
    obj: Any,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten nested JSON into key-value pairs.

    Lists are expanded using numeric indices.
    """

    items = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_json(v, new_key, sep=sep))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(v, new_key, sep=sep))

    else:
        items[parent_key] = obj

    return items


def object_contains_protein(obj: Any, protein_name: str) -> bool:
    """
    Check whether a JSON object contains a protein name anywhere.
    """

    protein_name_l = protein_name.lower()

    try:
        text = json.dumps(obj).lower()
    except Exception:
        text = str(obj).lower()

    pattern = rf"(^|[^a-z0-9]){re.escape(protein_name_l)}([^a-z0-9]|$)"
    return re.search(pattern, text) is not None


def collect_numeric_fields(flat: Dict[str, Any]) -> Dict[str, float]:
    """
    Collect numeric values from flattened JSON.
    """

    numeric = {}

    for k, v in flat.items():
        value = safe_float(v)
        if value is not None:
            numeric[sanitize_feature_name(k)] = value

    return numeric


# ============================================================
# UniProt retrieval
# ============================================================

def query_uniprot_for_protein(
    protein_name: str,
    organism_query: str = "bacteria",
    reviewed_first: bool = True,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Query UniProt REST API for a protein name.

    Because names like SrpR, PhlF, AmeR can be ambiguous, we query gene/protein terms
    and prefer reviewed entries when requested.

    Returns the first best matching entry.
    """

    base_url = "https://rest.uniprot.org/uniprotkb/search"

    # The query is intentionally broad but biased to bacteria.
    # You can edit organism_query to "Escherichia coli" or remove it.
    query = f'({protein_name}) AND ({organism_query})'

    params = {
        "query": query,
        "format": "json",
        "size": 10,
        "fields": "accession,id,protein_name,gene_names,organism_name,length,sequence,cc_function,go_p",
    }

    r = requests.get(base_url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", [])

    if not results:
        return None

    def score_entry(entry: Dict[str, Any]) -> int:
        score = 0
        text = json.dumps(entry).lower()
        pname = protein_name.lower()

        if re.search(rf"(^|[^a-z0-9]){re.escape(pname)}([^a-z0-9]|$)", text):
            score += 10

        if reviewed_first and entry.get("entryType", "").lower().startswith("uniprotkb reviewed"):
            score += 5

        # Prefer exact gene name if available
        genes = entry.get("genes", [])
        for g in genes:
            gene_name = g.get("geneName", {}).get("value", "")
            if gene_name.lower() == pname:
                score += 20

        return score

    results = sorted(results, key=score_entry, reverse=True)
    best = results[0]

    seq_obj = best.get("sequence", {})
    sequence = seq_obj.get("value", "")

    protein_desc = best.get("proteinDescription", {})
    recommended = protein_desc.get("recommendedName", {})
    full_name = ""
    if recommended:
        full_name = recommended.get("fullName", {}).get("value", "")

    genes = []
    for g in best.get("genes", []):
        if "geneName" in g:
            genes.append(g["geneName"].get("value", ""))
        for syn in g.get("synonyms", []):
            genes.append(syn.get("value", ""))

    organism = best.get("organism", {}).get("scientificName", "")

    comments = best.get("comments", [])
    function_texts = []
    for c in comments:
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts", []):
                function_texts.append(t.get("value", ""))

    return {
        "query_protein": protein_name,
        "uniprot_accession": best.get("primaryAccession", ""),
        "uniprot_id": best.get("uniProtkbId", ""),
        "uniprot_entry_type": best.get("entryType", ""),
        "uniprot_protein_name": full_name,
        "uniprot_gene_names": ";".join(sorted(set([g for g in genes if g]))),
        "uniprot_organism": organism,
        "uniprot_length": best.get("sequence", {}).get("length", np.nan),
        "uniprot_sequence": sequence,
        "uniprot_function": " ".join(function_texts),
    }


def build_uniprot_table(
    proteins: List[str],
    organism_query: str = "bacteria",
) -> pd.DataFrame:
    rows = []

    for p in tqdm(proteins, desc="Querying UniProt"):
        try:
            entry = query_uniprot_for_protein(p, organism_query=organism_query)
        except Exception as exc:
            entry = None
            print(f"[WARN] UniProt query failed for {p}: {exc}")

        if entry is None:
            entry = {
                "query_protein": p,
                "uniprot_accession": "",
                "uniprot_id": "",
                "uniprot_entry_type": "",
                "uniprot_protein_name": "",
                "uniprot_gene_names": "",
                "uniprot_organism": "",
                "uniprot_length": np.nan,
                "uniprot_sequence": "",
                "uniprot_function": "",
            }

        rows.append(entry)

    return pd.DataFrame(rows)


# ============================================================
# Physicochemical descriptors
# ============================================================

def compute_physicochemical_features(sequence: str) -> Dict[str, Any]:
    seq = clean_protein_sequence(sequence)

    features = {
        "seq_valid": int(len(seq) > 0),
        "seq_length": len(seq),
        "mw": np.nan,
        "aromaticity": np.nan,
        "instability_index": np.nan,
        "isoelectric_point": np.nan,
        "gravy": np.nan,
        "charge_at_pH_7": np.nan,
    }

    for aa in sorted(VALID_AA):
        features[f"aa_frac_{aa}"] = np.nan

    if len(seq) == 0:
        return features

    analysis = ProteinAnalysis(seq)

    features.update(
        {
            "mw": float(analysis.molecular_weight()),
            "aromaticity": float(analysis.aromaticity()),
            "instability_index": float(analysis.instability_index()),
            "isoelectric_point": float(analysis.isoelectric_point()),
            "gravy": float(analysis.gravy()),
            "charge_at_pH_7": float(analysis.charge_at_pH(7.0)),
        }
    )

    aa_percent = analysis.get_amino_acids_percent()

    for aa in sorted(VALID_AA):
        features[f"aa_frac_{aa}"] = float(aa_percent.get(aa, 0.0))

    return features


def add_physicochemical_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        seq = row.get("uniprot_sequence", "")
        rows.append(compute_physicochemical_features(seq))

    phys_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), phys_df.reset_index(drop=True)], axis=1)


# ============================================================
# Cello/UCF retrieval and parsing
# ============================================================

def download_cello_ucf_repo(
    output_dir: str,
    branch: str = "develop",
) -> str:
    """
    Download CIDARLAB/Cello-UCF repository ZIP and extract it.

    Default branch is 'develop' because the public README commonly points there.
    If this fails, try branch='master' or branch='main'.
    """

    url = f"https://github.com/CIDARLAB/Cello-UCF/archive/refs/heads/{branch}.zip"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / f"Cello-UCF-{branch}.zip"

    print(f"Downloading Cello-UCF from: {url}")

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    extracted_dirs = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith("Cello-UCF-")
    ]

    if not extracted_dirs:
        raise RuntimeError("Could not find extracted Cello-UCF directory.")

    return str(extracted_dirs[0])


def find_json_files(root_dir: str) -> List[str]:
    root = Path(root_dir)
    return [str(p) for p in root.rglob("*.json") if p.is_file()]


def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_objects(obj: Any) -> List[Any]:
    """
    Return all nested dict/list objects from a JSON structure.
    """

    found = []

    def visit(x):
        if isinstance(x, dict):
            found.append(x)
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)

    visit(obj)
    return found


def extract_ucf_records_for_protein(
    protein_name: str,
    json_files: List[str],
) -> pd.DataFrame:
    """
    Search all JSON files for objects mentioning the protein.
    Extract numeric fields and potential target metrics.
    """

    records = []

    for jf in json_files:
        try:
            obj = load_json_any(jf)
        except Exception:
            continue

        objects = iter_json_objects(obj)

        for idx, sub_obj in enumerate(objects):
            if not object_contains_protein(sub_obj, protein_name):
                continue

            flat = flatten_json(sub_obj)
            numeric = collect_numeric_fields(flat)

            text = json.dumps(sub_obj).lower()

            record = {
                "query_protein": protein_name,
                "ucf_json_file": jf,
                "ucf_object_index": idx,
                "ucf_object_type": clean_text(sub_obj.get("collection", sub_obj.get("type", sub_obj.get("class", "")))),
                "ucf_name": clean_text(sub_obj.get("name", sub_obj.get("gate_name", sub_obj.get("id", "")))),
                "ucf_contains_response_function": int(
                    "response" in text or "hill" in text or "ymax" in text or "ymin" in text
                ),
                "ucf_contains_snr": int("snr" in text or "signal_to_noise" in text or "signal-to-noise" in text),
            }

            # Add numeric fields with prefix
            for k, v in numeric.items():
                record[f"ucf_num_{k}"] = v

            # Try to derive target metrics
            target_info = derive_target_from_numeric_fields(numeric, flat)
            record.update(target_info)

            records.append(record)

    if not records:
        return pd.DataFrame(
            [
                {
                    "query_protein": protein_name,
                    "ucf_json_file": "",
                    "ucf_object_index": np.nan,
                    "ucf_object_type": "",
                    "ucf_name": "",
                    "ucf_contains_response_function": 0,
                    "ucf_contains_snr": 0,
                    "target_metric": np.nan,
                    "target_metric_name": "",
                    "target_metric_source": "not_found",
                }
            ]
        )

    return pd.DataFrame(records)


def derive_target_from_numeric_fields(
    numeric: Dict[str, float],
    flat: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Try to infer a target metric from UCF numeric fields.

    Priority:
    1. explicit SNR-like field;
    2. ymax/ymin ratio;
    3. dynamic range;
    4. explicit score/performance-like field;
    5. none.

    This function does not invent SNR. If SNR is not present, it uses an alternative
    target and labels it accordingly.
    """

    # Normalize names
    lower_to_original = {k.lower(): k for k in numeric.keys()}

    # 1. Explicit SNR
    for lk, orig in lower_to_original.items():
        if "snr" in lk or "signal_to_noise" in lk or "signal_noise" in lk:
            return {
                "target_metric": numeric[orig],
                "target_metric_name": "SNR",
                "target_metric_source": f"explicit_numeric_field:{orig}",
            }

    # 2. Find ymax and ymin
    ymax_key = None
    ymin_key = None

    for lk, orig in lower_to_original.items():
        if lk.endswith("ymax") or "ymax" == lk or "y_max" in lk:
            ymax_key = orig
        if lk.endswith("ymin") or "ymin" == lk or "y_min" in lk:
            ymin_key = orig

    if ymax_key is not None and ymin_key is not None:
        ymax = numeric[ymax_key]
        ymin = numeric[ymin_key]

        if ymin is not None and ymin > 0:
            on_off_ratio = ymax / ymin
            return {
                "target_metric": on_off_ratio,
                "target_metric_name": "on_off_ratio",
                "target_metric_source": f"derived_from:{ymax_key}/{ymin_key}",
                "derived_dynamic_range": ymax - ymin,
                "derived_log10_on_off_ratio": math.log10(on_off_ratio) if on_off_ratio > 0 else np.nan,
            }

        return {
            "target_metric": ymax - ymin,
            "target_metric_name": "dynamic_range",
            "target_metric_source": f"derived_from:{ymax_key}-{ymin_key}",
            "derived_dynamic_range": ymax - ymin,
            "derived_log10_on_off_ratio": np.nan,
        }

    # 3. Other performance-like fields
    performance_terms = [
        "score",
        "performance",
        "fitness",
        "accuracy",
        "dynamic_range",
        "on_off",
        "onoff",
        "fold_change",
        "foldchange",
    ]

    for lk, orig in lower_to_original.items():
        if any(term in lk for term in performance_terms):
            return {
                "target_metric": numeric[orig],
                "target_metric_name": orig,
                "target_metric_source": f"performance_like_numeric_field:{orig}",
            }

    return {
        "target_metric": np.nan,
        "target_metric_name": "",
        "target_metric_source": "not_found",
    }


def build_ucf_table(
    proteins: List[str],
    ucf_dir: Optional[str] = None,
    download_ucf: bool = True,
    output_cache_dir: str = "cache",
    branch: str = "develop",
) -> pd.DataFrame:
    """
    Build a table of Cello/UCF records associated with each protein.
    """

    if ucf_dir is None:
        if not download_ucf:
            raise ValueError("ucf_dir is None and download_ucf=False.")
        ucf_dir = download_cello_ucf_repo(output_cache_dir, branch=branch)

    json_files = find_json_files(ucf_dir)

    if not json_files:
        raise RuntimeError(f"No JSON files found under UCF directory: {ucf_dir}")

    print(f"Found {len(json_files)} JSON files in UCF directory.")

    dfs = []

    for p in tqdm(proteins, desc="Searching UCF records"):
        df_p = extract_ucf_records_for_protein(p, json_files)
        dfs.append(df_p)

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# Dataset construction
# ============================================================

def build_dataset_from_proteins(
    proteins: List[str],
    output_dir: str,
    organism_query: str = "bacteria",
    ucf_dir: Optional[str] = None,
    download_ucf: bool = True,
    ucf_branch: str = "develop",
    keep_only_rows_with_target: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build complete dataset.

    Returns:
    - full dataset;
    - dataset with target;
    - X;
    - y.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. UniProt
    uniprot_df = build_uniprot_table(proteins, organism_query=organism_query)
    uniprot_df.to_csv(output_dir / "uniprot_table.csv", index=False)

    # 2. Physicochemical features
    protein_df = add_physicochemical_features(uniprot_df)
    protein_df.to_csv(output_dir / "protein_features.csv", index=False)

    # 3. UCF/Cello features and targets
    ucf_df = build_ucf_table(
        proteins=proteins,
        ucf_dir=ucf_dir,
        download_ucf=download_ucf,
        output_cache_dir=str(output_dir / "cache"),
        branch=ucf_branch,
    )

    ucf_df.to_csv(output_dir / "ucf_records_raw.csv", index=False)

    # 4. Merge
    dataset = ucf_df.merge(protein_df, on="query_protein", how="left")

    # 5. Create an instance ID
    dataset.insert(
        0,
        "instance_id",
        [
            f"{row.query_protein}_{i}"
            for i, row in enumerate(dataset.itertuples(index=False))
        ],
    )

    # 6. Save full dataset
    dataset.to_csv(output_dir / "dataset_full.csv", index=False)

    # 7. Dataset with target
    dataset_with_target = dataset.copy()
    dataset_with_target["target_metric"] = pd.to_numeric(
        dataset_with_target["target_metric"],
        errors="coerce",
    )

    dataset_with_target = dataset_with_target.dropna(subset=["target_metric"])

    if keep_only_rows_with_target:
        dataset_model = dataset_with_target.copy()
    else:
        dataset_model = dataset.copy()

    dataset_with_target.to_csv(output_dir / "dataset_with_target.csv", index=False)

    # 8. Build X/y from rows with target only
    X, y = make_xy(dataset_with_target)

    X.to_csv(output_dir / "X.csv", index=False)
    y.to_csv(output_dir / "y.csv", index=False, header=True)

    return dataset, dataset_with_target, X, y


def make_xy(dataset_with_target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X/y for regression.

    Drops identifiers, raw sequences, raw text, paths and target columns.
    Encodes remaining categorical variables.
    """

    df = dataset_with_target.copy()

    y = pd.to_numeric(df["target_metric"], errors="coerce")

    drop_cols = [
        "instance_id",
        "query_protein",
        "ucf_json_file",
        "ucf_object_index",
        "uniprot_sequence",
        "uniprot_function",
        "target_metric",
        "target_metric_name",
        "target_metric_source",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove columns that are mostly raw long text
    for col in list(X.columns):
        if X[col].dtype == object:
            avg_len = X[col].astype(str).map(len).mean()
            if avg_len > 200:
                X = X.drop(columns=[col])

    X = pd.get_dummies(X, dummy_na=True)

    # Convert remaining to numeric where possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Simple imputation
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y


# ============================================================
# Optional: simple baseline regressor
# ============================================================

def train_baseline_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
) -> None:
    """
    Train a small baseline model only if there are enough rows.

    This is optional and mainly useful to verify that the dataset is usable.
    """

    if len(y) < 5:
        print("[INFO] Not enough target rows to train a baseline regressor.")
        return

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold

    n_splits = min(5, len(y))

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    result = pd.DataFrame(
        {
            "metric": ["MAE_mean", "MAE_std", "n_instances", "n_features"],
            "value": [
                -scores.mean(),
                scores.std(),
                len(y),
                X.shape[1],
            ],
        }
    )

    result.to_csv(Path(output_dir) / "baseline_cv_results.csv", index=False)

    print("\nBaseline RandomForestRegressor")
    print(result)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a synthetic biology regression dataset from protein names."
    )

    parser.add_argument(
        "--proteins",
        nargs="+",
        default=DEFAULT_PROTEINS,
        help="Protein names. Example: SrpR PhlF AmeR BetI QacR AmtR LmrA",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs_protein_dataset",
        help="Directory to save generated dataset files.",
    )

    parser.add_argument(
        "--organism-query",
        default="bacteria",
        help="UniProt query context. Example: bacteria, Escherichia coli",
    )

    parser.add_argument(
        "--ucf-dir",
        default=None,
        help="Optional local Cello-UCF directory. If omitted, the script downloads it.",
    )

    parser.add_argument(
        "--no-download-ucf",
        action="store_true",
        help="Disable automatic Cello-UCF download.",
    )

    parser.add_argument(
        "--ucf-branch",
        default="develop",
        help="Cello-UCF GitHub branch to download. Try develop, master or main.",
    )

    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Train a simple baseline regressor if enough target rows exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset, dataset_with_target, X, y = build_dataset_from_proteins(
        proteins=args.proteins,
        output_dir=args.output_dir,
        organism_query=args.organism_query,
        ucf_dir=args.ucf_dir,
        download_ucf=not args.no_download_ucf,
        ucf_branch=args.ucf_branch,
    )

    print("\nGenerated files:")
    print(f"- {args.output_dir}/uniprot_table.csv")
    print(f"- {args.output_dir}/protein_features.csv")
    print(f"- {args.output_dir}/ucf_records_raw.csv")
    print(f"- {args.output_dir}/dataset_full.csv")
    print(f"- {args.output_dir}/dataset_with_target.csv")
    print(f"- {args.output_dir}/X.csv")
    print(f"- {args.output_dir}/y.csv")

    print("\nSummary:")
    print(f"Full dataset shape: {dataset.shape}")
    print(f"Rows with target: {dataset_with_target.shape[0]}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if "target_metric_name" in dataset_with_target.columns:
        print("\nTarget metric distribution:")
        print(dataset_with_target["target_metric_name"].value_counts(dropna=False))

    if args.train_baseline:
        train_baseline_regressor(X, y, args.output_dir)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis


DEFAULT_PROTEINS = ["SrpR", "PhlF", "AmeR", "BetI", "QacR", "AmtR", "LmrA"]
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================
# Basic utilities
# ============================================================

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def sanitize_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def object_to_text(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False).lower()
    except Exception:
        return str(obj).lower()


def contains_protein(obj: Any, protein: str) -> bool:
    text = object_to_text(obj)
    p = protein.lower()
    return re.search(rf"(^|[^a-z0-9]){re.escape(p)}([^a-z0-9]|$)", text) is not None


def iter_dicts(obj: Any) -> List[Dict[str, Any]]:
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


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_json(v, new_key))
    else:
        out[prefix] = obj

    return out


# ============================================================
# Download / locate Cello-UCF
# ============================================================

def download_cello_ucf(output_dir: str, branch: str = "develop") -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://github.com/CIDARLAB/Cello-UCF/archive/refs/heads/{branch}.zip"
    zip_path = output_dir / f"Cello-UCF-{branch}.zip"

    print(f"Downloading Cello-UCF: {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    candidates = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith("Cello-UCF-")
    ]

    if not candidates:
        raise RuntimeError("Could not locate extracted Cello-UCF directory.")

    return str(candidates[0])


def find_json_files(root_dir: str) -> List[str]:
    return [
        str(p) for p in Path(root_dir).rglob("*.json")
        if p.is_file()
    ]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Robust parameter extraction from UCF JSON
# ============================================================

def extract_named_parameters(obj: Any) -> Dict[str, float]:
    """
    Extract parameters from UCF-like JSON structures.

    Handles cases such as:
    {"name": "ymax", "value": 5.12}
    {"parameter": "ymin", "value": 0.01}
    {"var": "K", "num": 0.15}
    {"parameters": [{"name": "n", "value": 2.4}]}

    Also handles direct keys:
    {"ymax": 5.12, "ymin": 0.01}
    """

    params = {}

    candidate_name_keys = [
        "name", "parameter", "param", "id", "key", "var", "variable"
    ]

    candidate_value_keys = [
        "value", "val", "num", "default", "default_value", "mean", "median"
    ]

    def visit(x):
        if isinstance(x, dict):
            # Direct fields: ymax, ymin, K, n, alpha, beta, etc.
            for k, v in x.items():
                lk = str(k).lower()
                if lk in {
                    "ymax", "y_max", "ymin", "y_min",
                    "k", "kd", "km", "n", "hill", "hill_coefficient",
                    "alpha", "beta", "snr", "signal_to_noise",
                    "score", "fitness", "dynamic_range", "on_off_ratio",
                }:
                    fv = safe_float(v)
                    if fv is not None:
                        params[lk] = fv

            # Name/value parameter objects
            found_name = None
            found_value = None

            for nk in candidate_name_keys:
                if nk in x:
                    found_name = str(x[nk]).strip().lower()
                    break

            for vk in candidate_value_keys:
                if vk in x:
                    found_value = safe_float(x[vk])
                    if found_value is not None:
                        break

            if found_name and found_value is not None:
                params[found_name] = found_value

            for v in x.values():
                visit(v)

        elif isinstance(x, list):
            for v in x:
                visit(v)

    visit(obj)

    # Normalize synonyms
    normalized = {}

    for k, v in params.items():
        kk = k.lower().strip()
        kk = kk.replace("-", "_").replace(" ", "_")

        if kk in {"y_max", "ymax"}:
            normalized["ymax"] = v
        elif kk in {"y_min", "ymin"}:
            normalized["ymin"] = v
        elif kk in {"hill", "hill_coefficient", "n"}:
            normalized["n"] = v
        elif kk in {"k", "kd", "km"}:
            normalized["K"] = v
        elif kk in {"snr", "signal_to_noise", "signal_noise_ratio"}:
            normalized["SNR"] = v
        elif kk in {"on_off", "onoff", "on_off_ratio"}:
            normalized["on_off_ratio"] = v
        elif kk in {"dynamic_range", "range"}:
            normalized["dynamic_range"] = v
        else:
            normalized[sanitize_name(kk)] = v

    return normalized


def infer_gate_name(obj: Dict[str, Any], protein: str) -> str:
    """
    Try to infer a gate/circuit-fragment name from common UCF fields.
    """

    candidate_keys = [
        "gate_name", "name", "id", "group_name", "regulator",
        "system", "device", "part_name"
    ]

    for k in candidate_keys:
        if k in obj and obj[k] is not None:
            value = str(obj[k])
            if protein.lower() in value.lower():
                return value

    text = object_to_text(obj)

    # Try patterns like P1_PhlF, Q1_QacR, A1_AmtR
    m = re.search(rf"[A-Za-z0-9]+_{re.escape(protein)}[A-Za-z0-9]*", text, flags=re.I)
    if m:
        return m.group(0)

    return protein


def infer_record_type(obj: Dict[str, Any]) -> str:
    for k in ["collection", "type", "class", "category"]:
        if k in obj:
            return str(obj[k])
    return ""


def derive_target(params: Dict[str, float]) -> Dict[str, Any]:
    """
    Target priority:
    1. SNR if explicit
    2. on_off_ratio if explicit
    3. ymax/ymin
    4. dynamic_range
    """

    if "SNR" in params:
        return {
            "target_metric": params["SNR"],
            "target_metric_name": "SNR",
            "target_metric_source": "explicit_UCF_or_JSON_parameter",
        }

    if "on_off_ratio" in params:
        return {
            "target_metric": params["on_off_ratio"],
            "target_metric_name": "on_off_ratio",
            "target_metric_source": "explicit_UCF_or_JSON_parameter",
        }

    ymax = params.get("ymax")
    ymin = params.get("ymin")

    if ymax is not None and ymin is not None:
        dynamic_range = ymax - ymin

        if ymin > 0:
            on_off_ratio = ymax / ymin
            return {
                "target_metric": on_off_ratio,
                "target_metric_name": "on_off_ratio",
                "target_metric_source": "derived_from_ymax_ymin",
                "derived_dynamic_range": dynamic_range,
                "derived_log10_on_off_ratio": math.log10(on_off_ratio)
                if on_off_ratio > 0 else np.nan,
            }

        return {
            "target_metric": dynamic_range,
            "target_metric_name": "dynamic_range",
            "target_metric_source": "derived_from_ymax_ymin",
            "derived_dynamic_range": dynamic_range,
            "derived_log10_on_off_ratio": np.nan,
        }

    if "dynamic_range" in params:
        return {
            "target_metric": params["dynamic_range"],
            "target_metric_name": "dynamic_range",
            "target_metric_source": "explicit_UCF_or_JSON_parameter",
        }

    return {
        "target_metric": np.nan,
        "target_metric_name": "",
        "target_metric_source": "not_found",
        "derived_dynamic_range": np.nan,
        "derived_log10_on_off_ratio": np.nan,
    }


def extract_ucf_gate_records_for_protein(
    protein: str,
    json_files: List[str],
) -> pd.DataFrame:
    """
    Build instances representing gates/circuit fragments associated with a protein.
    """

    rows = []

    for jf in json_files:
        try:
            data = load_json(jf)
        except Exception:
            continue

        for idx, obj in enumerate(iter_dicts(data)):
            if not contains_protein(obj, protein):
                continue

            params = extract_named_parameters(obj)

            # Keep only objects that look informative.
            has_response_params = any(k in params for k in ["ymax", "ymin", "K", "n", "SNR", "on_off_ratio"])
            text = object_to_text(obj)
            looks_like_gate = (
                "gate" in text
                or "response" in text
                or "hill" in text
                or "ymax" in text
                or "ymin" in text
                or "regulator" in text
            )

            if not has_response_params and not looks_like_gate:
                continue

            target = derive_target(params)

            row = {
                "query_protein": protein,
                "instance_type": "ucf_gate_or_circuit_fragment",
                "ucf_json_file": jf,
                "ucf_object_index": idx,
                "ucf_record_type": infer_record_type(obj),
                "gate_or_fragment_name": infer_gate_name(obj, protein),
                "has_response_parameters": int(has_response_params),
            }

            for k, v in params.items():
                row[f"ucf_param_{sanitize_name(k)}"] = v

            row.update(target)
            rows.append(row)

    if not rows:
        return pd.DataFrame([{
            "query_protein": protein,
            "instance_type": "not_found",
            "ucf_json_file": "",
            "ucf_object_index": np.nan,
            "ucf_record_type": "",
            "gate_or_fragment_name": protein,
            "has_response_parameters": 0,
            "target_metric": np.nan,
            "target_metric_name": "",
            "target_metric_source": "not_found",
            "derived_dynamic_range": np.nan,
            "derived_log10_on_off_ratio": np.nan,
        }])

    df = pd.DataFrame(rows)

    # Remove exact duplicate records that arise from nested JSON traversal.
    subset_cols = [
        "query_protein",
        "gate_or_fragment_name",
        "ucf_param_ymax",
        "ucf_param_ymin",
        "ucf_param_K",
        "ucf_param_n",
        "target_metric",
        "target_metric_name",
    ]

    subset_cols = [c for c in subset_cols if c in df.columns]
    df = df.drop_duplicates(subset=subset_cols)

    return df


# ============================================================
# UniProt + physicochemical features
# ============================================================

def query_uniprot(protein: str, organism_query: str = "bacteria") -> Dict[str, Any]:
    url = "https://rest.uniprot.org/uniprotkb/search"

    params = {
        "query": f"({protein}) AND ({organism_query})",
        "format": "json",
        "size": 10,
        "fields": "accession,id,protein_name,gene_names,organism_name,length,sequence,cc_function",
    }

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"[WARN] UniProt failed for {protein}: {exc}")
        return empty_uniprot_record(protein)

    results = data.get("results", [])

    if not results:
        return empty_uniprot_record(protein)

    def score(entry):
        s = 0
        text = object_to_text(entry)
        p = protein.lower()

        if re.search(rf"(^|[^a-z0-9]){re.escape(p)}([^a-z0-9]|$)", text):
            s += 10

        for g in entry.get("genes", []):
            gn = g.get("geneName", {}).get("value", "")
            if gn.lower() == p:
                s += 20

        if entry.get("entryType", "").lower().startswith("uniprotkb reviewed"):
            s += 5

        return s

    best = sorted(results, key=score, reverse=True)[0]

    seq = best.get("sequence", {}).get("value", "")
    organism = best.get("organism", {}).get("scientificName", "")

    protein_name = ""
    try:
        protein_name = (
            best.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )
    except Exception:
        pass

    genes = []
    for g in best.get("genes", []):
        if "geneName" in g:
            genes.append(g["geneName"].get("value", ""))
        for syn in g.get("synonyms", []):
            genes.append(syn.get("value", ""))

    function_text = []
    for c in best.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts", []):
                function_text.append(t.get("value", ""))

    return {
        "query_protein": protein,
        "uniprot_accession": best.get("primaryAccession", ""),
        "uniprot_id": best.get("uniProtkbId", ""),
        "uniprot_entry_type": best.get("entryType", ""),
        "uniprot_protein_name": protein_name,
        "uniprot_gene_names": ";".join(sorted(set([g for g in genes if g]))),
        "uniprot_organism": organism,
        "uniprot_length": best.get("sequence", {}).get("length", np.nan),
        "uniprot_sequence": seq,
        "uniprot_function": " ".join(function_text),
    }


def empty_uniprot_record(protein: str) -> Dict[str, Any]:
    return {
        "query_protein": protein,
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


def clean_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return "".join([aa for aa in seq if aa in VALID_AA])


def physicochemical_features(seq: str) -> Dict[str, Any]:
    seq = clean_sequence(seq)

    out = {
        "seq_valid": int(len(seq) > 0),
        "seq_length": len(seq),
        "protein_mw": np.nan,
        "protein_aromaticity": np.nan,
        "protein_instability_index": np.nan,
        "protein_isoelectric_point": np.nan,
        "protein_gravy": np.nan,
        "protein_charge_at_pH_7": np.nan,
    }

    for aa in sorted(VALID_AA):
        out[f"protein_aa_frac_{aa}"] = np.nan

    if not seq:
        return out

    analysis = ProteinAnalysis(seq)

    out.update({
        "protein_mw": float(analysis.molecular_weight()),
        "protein_aromaticity": float(analysis.aromaticity()),
        "protein_instability_index": float(analysis.instability_index()),
        "protein_isoelectric_point": float(analysis.isoelectric_point()),
        "protein_gravy": float(analysis.gravy()),
        "protein_charge_at_pH_7": float(analysis.charge_at_pH(7.0)),
    })

    aa = analysis.get_amino_acids_percent()
    for k in sorted(VALID_AA):
        out[f"protein_aa_frac_{k}"] = float(aa.get(k, 0.0))

    return out


def build_protein_feature_table(proteins: List[str], organism_query: str) -> pd.DataFrame:
    rows = []

    for p in tqdm(proteins, desc="UniProt + protein features"):
        rec = query_uniprot(p, organism_query=organism_query)
        rec.update(physicochemical_features(rec.get("uniprot_sequence", "")))
        rows.append(rec)

    return pd.DataFrame(rows)


# ============================================================
# Dataset builder
# ============================================================

def build_dataset(
    proteins: List[str],
    output_dir: str,
    ucf_dir: Optional[str],
    download_ucf_flag: bool,
    ucf_branch: str,
    organism_query: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ucf_dir is None:
        if not download_ucf_flag:
            raise ValueError("Provide --ucf-dir or allow automatic download.")
        ucf_dir = download_cello_ucf(str(output_dir / "cache"), branch=ucf_branch)

    json_files = find_json_files(ucf_dir)
    print(f"Found {len(json_files)} JSON files.")

    # UCF gate/circuit-fragment records
    ucf_tables = []

    for p in tqdm(proteins, desc="Extracting UCF gate records"):
        ucf_tables.append(extract_ucf_gate_records_for_protein(p, json_files))

    ucf_df = pd.concat(ucf_tables, ignore_index=True)
    ucf_df.to_csv(output_dir / "ucf_gate_records.csv", index=False)

    # Protein features
    protein_df = build_protein_feature_table(proteins, organism_query)
    protein_df.to_csv(output_dir / "protein_features.csv", index=False)

    # Merge
    dataset = ucf_df.merge(protein_df, on="query_protein", how="left")

    dataset.insert(
        0,
        "instance_id",
        [
            f"{row.query_protein}_{sanitize_name(str(row.gate_or_fragment_name))}_{i}"
            for i, row in enumerate(dataset.itertuples(index=False))
        ],
    )

    dataset.to_csv(output_dir / "dataset_full.csv", index=False)

    dataset_with_target = dataset.copy()
    dataset_with_target["target_metric"] = pd.to_numeric(
        dataset_with_target["target_metric"],
        errors="coerce"
    )
    dataset_with_target = dataset_with_target.dropna(subset=["target_metric"])
    dataset_with_target.to_csv(output_dir / "dataset_with_target.csv", index=False)

    X, y = make_xy(dataset_with_target)
    X.to_csv(output_dir / "X.csv", index=False)
    y.to_csv(output_dir / "y.csv", index=False, header=True)

    return dataset, dataset_with_target, X, y


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df["target_metric"], errors="coerce")

    drop_cols = [
        "instance_id",
        "query_protein",
        "ucf_json_file",
        "ucf_object_index",
        "target_metric",
        "target_metric_name",
        "target_metric_source",
        "uniprot_sequence",
        "uniprot_function",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop very long text columns
    for col in list(X.columns):
        if X[col].dtype == object:
            avg_len = X[col].astype(str).map(len).mean()
            if avg_len > 150:
                X = X.drop(columns=[col])

    X = pd.get_dummies(X, dummy_na=True)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proteins",
        nargs="+",
        default=DEFAULT_PROTEINS,
        help="Protein names."
    )

    parser.add_argument(
        "--output-dir",
        default="outputs_circuit_dataset",
        help="Output directory."
    )

    parser.add_argument(
        "--ucf-dir",
        default=None,
        help="Local Cello-UCF directory. If omitted, the script downloads it."
    )

    parser.add_argument(
        "--no-download-ucf",
        action="store_true",
        help="Disable automatic Cello-UCF download."
    )

    parser.add_argument(
        "--ucf-branch",
        default="develop",
        help="Cello-UCF branch. Try develop, master, or main."
    )

    parser.add_argument(
        "--organism-query",
        default="bacteria",
        help="UniProt organism query context."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset, dataset_with_target, X, y = build_dataset(
        proteins=args.proteins,
        output_dir=args.output_dir,
        ucf_dir=args.ucf_dir,
        download_ucf_flag=not args.no_download_ucf,
        ucf_branch=args.ucf_branch,
        organism_query=args.organism_query,
    )

    print("\nFiles generated:")
    print(f"- {args.output_dir}/ucf_gate_records.csv")
    print(f"- {args.output_dir}/protein_features.csv")
    print(f"- {args.output_dir}/dataset_full.csv")
    print(f"- {args.output_dir}/dataset_with_target.csv")
    print(f"- {args.output_dir}/X.csv")
    print(f"- {args.output_dir}/y.csv")

    print("\nSummary:")
    print(f"Full dataset shape: {dataset.shape}")
    print(f"Rows with target: {dataset_with_target.shape[0]}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if len(dataset_with_target) > 0:
        print("\nTarget metric distribution:")
        print(dataset_with_target["target_metric_name"].value_counts(dropna=False))

        cols = [
            "query_protein",
            "gate_or_fragment_name",
            "ucf_param_ymax",
            "ucf_param_ymin",
            "ucf_param_K",
            "ucf_param_n",
            "target_metric",
            "target_metric_name",
            "target_metric_source",
        ]
        cols = [c for c in cols if c in dataset_with_target.columns]

        print("\nPreview with target:")
        print(dataset_with_target[cols].head(20))
    else:
        print("\nNo target found.")
        print("Check outputs_circuit_dataset/ucf_gate_records.csv to inspect extracted fields.")


if __name__ == "__main__":
    main()

'''
     An ON/OFF ratio in biological circuits (specifically synthetic biology) measures the performance of a genetic 
     switch. It is calculated by dividing the maximum desired output (the ON state, often measured by fluorescence 
    or protein production) by the minimum leakage (the OFF state)
'''
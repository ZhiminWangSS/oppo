#!/usr/bin/env python3
"""Consolidate experimental results from JSON files into a single Excel workbook.

Usage:
    python consolidate_results_to_excel.py

The script searches subfolders named 0-23 within the base directory defined below.
If a "result_episode.json" file is present, it reads the JSON content, flattens it
(using pandas.json_normalize if the data is dict/list), adds a column indicating
its source folder, and appends the data to an aggregate DataFrame. Finally, the
DataFrame is written to an Excel file named "results_summary.xlsx" in the base
results directory.
"""
import json
from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    sys.stderr.write("pandas is required. Install via `pip install pandas` and retry.\n")
    sys.exit(1)

def load_json(path: Path):
    """Load JSON content safely, returning None on failure."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        sys.stderr.write(f"Failed to read {path}: {exc}\n")
        return None

def main():
    base_dir = Path("/home/aiseon/zmwang/codes/coela/oppo/tdw_mat/results_cobelv1.2_0828/LMs-deepseek-chat/run_20250829_1735")
    if not base_dir.exists():
        sys.stderr.write(f"Base directory not found: {base_dir}\n")
        sys.exit(1)

    all_rows = []

    for folder_idx in range(24):
        sub_dir = base_dir / str(folder_idx)
        result_file = sub_dir / "result_episode.json"
        if not result_file.exists():
            # Skip missing files
            continue
        data = load_json(result_file)
        if data is None:
            continue

        # Normalize depending on the JSON structure
        if isinstance(data, list):
            df_part = pd.json_normalize(data)
        elif isinstance(data, dict):
            df_part = pd.json_normalize(data)
        else:
            # Unsupported structure; skip
            sys.stderr.write(f"Unsupported JSON structure in {result_file}\n")
            continue

        # Add source folder column
        df_part.insert(0, "source_folder", folder_idx)
        all_rows.append(df_part)

    if not all_rows:
        sys.stderr.write("No result_episode.json files found.\n")
        sys.exit(1)

    df = pd.concat(all_rows, ignore_index=True)

    output_path = base_dir / "results_summary.xlsx"
    try:
        df.to_excel(output_path, index=False)
        print(f"Excel summary created at: {output_path}")
    except Exception as exc:
        sys.stderr.write(f"Failed to write Excel file: {exc}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm


def parse_folder_name(folder_name):
    """
    Expect format: q{number}_shots{number}
    Example: q4_shots128
    """
    m = re.match(r"q(\d+)_shots(\d+)", folder_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def estimate_eps_from_gradvar(df):
    """
    Estimate epsilon from:
        log(avg_grad_var_lastK) = a - eps * L
    """
    df = df.copy()

    # Keep only positive gradient variance
    df = df[df["avg_grad_var_lastK"] > 0]

    if len(df) < 3:
        return None

    y = np.log(df["avg_grad_var_lastK"])
    X = sm.add_constant(df["cfg_L"])

    model = sm.OLS(y, X).fit()

    eps_hat = -model.params["cfg_L"]

    return {
        "eps_hat": float(eps_hat),
        "r2": float(model.rsquared),
        "n_points": int(len(df)),
        "p_value_L": float(model.pvalues["cfg_L"])
    }


def main(input_path):

    results = []

    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)

        if not os.path.isdir(folder_path):
            continue

        q, shots = parse_folder_name(folder)
        if q is None:
            continue

        csv_path = os.path.join(folder_path, "deltas_per_seed.csv")

        if not os.path.exists(csv_path):
            print(f"[SKIP] {folder} -> deltas_per_seed.csv not found")
            continue

        print(f"[PROCESSING] {folder}")

        df = pd.read_csv(csv_path)

        required_cols = {"cfg_L", "avg_grad_var_lastK"}
        if not required_cols.issubset(df.columns):
            print(f"[SKIP] {folder} -> missing required columns")
            continue

        est = estimate_eps_from_gradvar(df)

        if est is None:
            print(f"[SKIP] {folder} -> not enough valid data")
            continue

        est.update({
            "q": q,
            "shots": shots
        })

        results.append(est)

    if len(results) == 0:
        print("No epsilon estimates computed.")
        return

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(input_path, "epsilon_estimates.csv")
    out_json = os.path.join(input_path, "epsilon_estimates.json")

    out_df.to_csv(out_csv, index=False)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved results:")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Root folder containing q*_shots* subfolders"
    )
    args = parser.parse_args()
    main(args.input)
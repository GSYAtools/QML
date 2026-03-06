#!/usr/bin/env python3
"""
analyze_experiment_regimes_batch.py

Versión batch corregida:
 - evita error de linregress cuando todos los x son idénticos
 - etiquetas en español y SIN títulos en las figuras
 - guarda subcarpetas por combinación (q, shots) como antes
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- Utilities ----------------
def safe_to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------------- Stats helpers ----------------
def bootstrap_paired_diff_ci(a, b, n_boot=5000, alpha=0.05, rng_seed=123456):
    rng = np.random.default_rng(rng_seed)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = a.size
    if n == 0:
        return np.nan, (np.nan, np.nan), n
    diffs = a - b
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    lo = float(np.percentile(boots, 100*(alpha/2)))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return float(diffs.mean()), (lo, hi), int(n)

def cohens_d_paired(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    nd = len(d)
    if nd <= 1:
        return np.nan
    sd = d.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(d.mean() / sd)

def paired_ttest(x, y):
    try:
        t, p = stats.ttest_rel(x, y, nan_policy='omit')
        return float(t), float(p)
    except Exception:
        return np.nan, np.nan

def adjust_pvals_bh(pvals):
    p = np.asarray([1.0 if (p is None or (isinstance(p,float) and math.isnan(p))) else p for p in pvals], dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    idx = np.argsort(p)
    sorted_p = p[idx]
    bh = np.empty(m, dtype=float)
    for i, pv in enumerate(sorted_p, start=1):
        bh[i-1] = pv * m / i
    # enforce monotonicity
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.minimum(bh, 1.0)
    adj = np.empty(m, dtype=float)
    adj[idx] = bh
    return adj

# ---------------- Core analysis (adapted) ----------------
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    numeric_cols = [
        "final_acc","final_loss","avg_grad_var_lastK","cfg_q","cfg_L","cfg_shots",
        "n_actions_total","n_actions_shots_doubled","n_actions_inject"
    ]
    df = safe_to_numeric(df, numeric_cols)
    return df

def compute_snr_proxy(df):
    if "avg_grad_var_lastK" in df.columns:
        df["snr_proxy_invvar"] = df["avg_grad_var_lastK"].apply(lambda x: 1.0/np.sqrt(x) if (pd.notna(x) and x>0) else np.nan)
        df["snr_proxy_inv"] = df["avg_grad_var_lastK"].apply(lambda x: 1.0/x if (pd.notna(x) and x>0) else np.nan)
    else:
        df["snr_proxy_invvar"] = np.nan
        df["snr_proxy_inv"] = np.nan

    if "mean_abs_grad" in df.columns and "avg_grad_var_lastK" in df.columns:
        df["snr_from_mean"] = df.apply(lambda r: (r["mean_abs_grad"] / np.sqrt(r["avg_grad_var_lastK"]))
                                       if pd.notna(r["mean_abs_grad"]) and pd.notna(r["avg_grad_var_lastK"]) and r["avg_grad_var_lastK"]>0 else np.nan,
                                       axis=1)
    else:
        df["snr_from_mean"] = np.nan
    return df

def build_pivot_and_deltas(df):
    pivot = df.pivot_table(index=["run_id","seed","cfg_L","cfg_shots","cfg_q"], columns="mode", values="final_acc", aggfunc="first")
    act_pivot_shots = df.pivot_table(index=["run_id","seed","cfg_L","cfg_shots","cfg_q"], columns="mode", values="n_actions_shots_doubled", aggfunc="first")
    act_pivot_inject = df.pivot_table(index=["run_id","seed","cfg_L","cfg_shots","cfg_q"], columns="mode", values="n_actions_inject", aggfunc="first")
    grad_proxy = df.groupby(["run_id","seed","cfg_L","cfg_shots","cfg_q"])["avg_grad_var_lastK"].first()
    snr_inv = df.groupby(["run_id","seed","cfg_L","cfg_shots","cfg_q"])["snr_proxy_invvar"].first()
    snr_inv2 = df.groupby(["run_id","seed","cfg_L","cfg_shots","cfg_q"])["snr_proxy_inv"].first()
    snr_mean = df.groupby(["run_id","seed","cfg_L","cfg_shots","cfg_q"])["snr_from_mean"].first()

    pivot = pivot.reset_index()
    rows = []
    # iterate pivot rows
    for _, row in pivot.iterrows():
        key = (row["run_id"], row["seed"], row["cfg_L"], row["cfg_shots"], row["cfg_q"])
        baseline = row.get("baseline_noisy", np.nan)
        shots_acc = row.get("OF_noisy_shots", np.nan)
        inject_acc = row.get("OF_noisy_inject", np.nan)
        # actions
        try:
            n_shots = act_pivot_shots.loc[key].get("OF_noisy_shots", np.nan) if key in act_pivot_shots.index else np.nan
        except Exception:
            n_shots = np.nan
        try:
            n_inject = act_pivot_inject.loc[key].get("OF_noisy_inject", np.nan) if key in act_pivot_inject.index else np.nan
        except Exception:
            n_inject = np.nan
        # proxies
        try:
            gradv = grad_proxy.loc[key] if key in grad_proxy.index else np.nan
        except Exception:
            gradv = np.nan
        try:
            snr1 = snr_inv.loc[key] if key in snr_inv.index else np.nan
        except Exception:
            snr1 = np.nan
        try:
            snr2 = snr_inv2.loc[key] if key in snr_inv2.index else np.nan
        except Exception:
            snr2 = np.nan
        try:
            snrm = snr_mean.loc[key] if key in snr_mean.index else np.nan
        except Exception:
            snrm = np.nan

        rows.append({
            "run_id": row["run_id"],
            "seed": row["seed"],
            "cfg_L": row["cfg_L"],
            "cfg_shots": row["cfg_shots"],
            "cfg_q": row["cfg_q"],
            "final_acc_baseline": baseline,
            "final_acc_shots": shots_acc,
            "final_acc_inject": inject_acc,
            "delta_shots": (shots_acc - baseline) if (pd.notna(shots_acc) and pd.notna(baseline)) else np.nan,
            "delta_inject": (inject_acc - baseline) if (pd.notna(inject_acc) and pd.notna(baseline)) else np.nan,
            "n_actions_shots": n_shots,
            "n_actions_inject": n_inject,
            "avg_grad_var_lastK": gradv,
            "snr_proxy_invvar": snr1,
            "snr_proxy_inv": snr2,
            "snr_from_mean": snrm
        })
    deltas_df = pd.DataFrame(rows)
    return pivot, deltas_df

def aggregate_and_stats(deltas_df, outdir, n_boot=5000):
    ensure_dir(outdir)
    results = []
    Ls = sorted(deltas_df['cfg_L'].dropna().unique().tolist())
    for L in Ls:
        sub = deltas_df[deltas_df['cfg_L'] == L]
        mean_shots, (lo_s, hi_s), n_sh = bootstrap_paired_diff_ci(sub['final_acc_shots'].values, sub['final_acc_baseline'].values, n_boot=n_boot)
        t_s, p_s = paired_ttest(sub['final_acc_shots'].values, sub['final_acc_baseline'].values)
        d_s = cohens_d_paired(sub['final_acc_shots'].values, sub['final_acc_baseline'].values)
        mean_inj, (lo_i, hi_i), n_i = bootstrap_paired_diff_ci(sub['final_acc_inject'].values, sub['final_acc_baseline'].values, n_boot=n_boot)
        t_i, p_i = paired_ttest(sub['final_acc_inject'].values, sub['final_acc_baseline'].values)
        d_i = cohens_d_paired(sub['final_acc_inject'].values, sub['final_acc_baseline'].values)
        results.append({
            "cfg_L": L,
            "n_pairs_shots": int(n_sh),
            "mean_delta_shots": float(mean_shots) if not np.isnan(mean_shots) else np.nan,
            "ci_lo_shots": float(lo_s) if not np.isnan(lo_s) else np.nan,
            "ci_hi_shots": float(hi_s) if not np.isnan(hi_s) else np.nan,
            "t_shots": float(t_s) if not np.isnan(t_s) else np.nan,
            "p_shots": float(p_s) if not np.isnan(p_s) else np.nan,
            "cohen_d_shots": float(d_s) if not np.isnan(d_s) else np.nan,
            "n_pairs_inject": int(n_i),
            "mean_delta_inject": float(mean_inj) if not np.isnan(mean_inj) else np.nan,
            "ci_lo_inject": float(lo_i) if not np.isnan(lo_i) else np.nan,
            "ci_hi_inject": float(hi_i) if not np.isnan(hi_i) else np.nan,
            "t_inject": float(t_i) if not np.isnan(t_i) else np.nan,
            "p_inject": float(p_i) if not np.isnan(p_i) else np.nan,
            "cohen_d_inject": float(d_i) if not np.isnan(d_i) else np.nan,
            "median_grad_var_lastK": float(sub['avg_grad_var_lastK'].median()) if len(sub)>0 else np.nan,
            "mean_snr_proxy": float(sub['snr_proxy_invvar'].mean()) if 'snr_proxy_invvar' in sub.columns else np.nan,
            "n_seeds_total": int(len(sub))
        })
    stats_df = pd.DataFrame(results)
    p_shots = stats_df['p_shots'].tolist()
    p_inject = stats_df['p_inject'].tolist()
    stats_df['p_shots_fdr'] = adjust_pvals_bh(p_shots)
    stats_df['p_inject_fdr'] = adjust_pvals_bh(p_inject)
    stats_df.to_csv(Path(outdir)/"stats_by_L.csv", index=False)
    return stats_df

# ---------------- Plots (en español y sin títulos) ----------------
def plot_delta_vs_L(stats_df, deltas_df, outpath, title_extra=""):
    Ls = stats_df['cfg_L'].values
    mean_shots = stats_df['mean_delta_shots'].values
    mean_inj = stats_df['mean_delta_inject'].values
    err_lower_shots = mean_shots - stats_df['ci_lo_shots'].values
    err_upper_shots = stats_df['ci_hi_shots'].values - mean_shots
    err_lower_inj = mean_inj - stats_df['ci_lo_inject'].values
    err_upper_inj = stats_df['ci_hi_inject'].values - mean_inj
    plt.figure(figsize=(7,4))
    plt.errorbar(Ls, mean_shots, yerr=[err_lower_shots, err_upper_shots], marker='o', label='Δ acc (OF_shots)')
    plt.errorbar(Ls, mean_inj, yerr=[err_lower_inj, err_upper_inj], marker='o', label='Δ acc (OF_inject)')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("L (n_layers)")
    plt.ylabel("Δ accuracy vs baseline_noisy (em pares)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_gradvar_by_L(deltas_df, outpath, title_extra=""):
    plt.figure(figsize=(8,4))
    Ls = sorted(deltas_df['cfg_L'].dropna().unique().tolist())
    means = []
    for L in Ls:
        sub = deltas_df[deltas_df['cfg_L']==L]
        means.append(sub['avg_grad_var_lastK'].mean() if len(sub)>0 else np.nan)
    plt.plot(Ls, means, marker='o', label='avg_grad_var_lastK (media)')
    plt.xlabel("L (n_layers)")
    plt.ylabel("avg_grad_var_lastK")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def heatmap_L_shots(deltas_df, outpath_prefix):
    Ls = sorted(deltas_df['cfg_L'].dropna().unique().tolist())
    shots = sorted(deltas_df['cfg_shots'].dropna().unique().tolist())
    def pivot_metric(metric):
        table = deltas_df.groupby(['cfg_L','cfg_shots'])[metric].median().unstack(fill_value=np.nan)
        return table.reindex(index=Ls, columns=shots)
    for metric, name in [("delta_shots","delta_shots"), ("delta_inject","delta_inject")]:
        table = pivot_metric(metric)
        plt.figure(figsize=(6,4))
        plt.imshow(table.values, aspect='auto', origin='lower')
        plt.colorbar(label=name)
        plt.xticks(range(len(table.columns)), table.columns, rotation=45)
        plt.yticks(range(len(table.index)), table.index)
        plt.xlabel("shots")
        plt.ylabel("L")
        plt.tight_layout()
        plt.savefig(f"{outpath_prefix}_{name}.png", dpi=200)
        plt.close()
    table_sh = pivot_metric("delta_shots")
    table_in = pivot_metric("delta_inject")
    diff = table_in - table_sh
    plt.figure(figsize=(6,4))
    vmax = np.nanmax(np.abs(diff.values)) if not np.isnan(np.nanmax(np.abs(diff.values))) else 1.0
    plt.imshow(diff.values, aspect='auto', origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(label="delta_inject - delta_shots")
    plt.xticks(range(len(diff.columns)), diff.columns, rotation=45)
    plt.yticks(range(len(diff.index)), diff.index)
    plt.xlabel("shots")
    plt.ylabel("L")
    plt.tight_layout()
    plt.savefig(f"{outpath_prefix}_delta_diff.png", dpi=200)
    plt.close()

def scatter_actions_vs_delta(deltas_df, outpath_prefix):
    # Shots-actions vs delta
    x = deltas_df['n_actions_shots'].values
    y = deltas_df['delta_shots'].values
    mask = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(6,4))
    plt.scatter(x[mask], y[mask])
    # only attempt regression if there is variation in x and at least 3 points
    if mask.sum() > 2 and np.nanstd(x[mask]) > 0 and len(np.unique(x[mask])) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 50)
        plt.plot(xs, intercept + slope*xs, color='C1', label=f"OLS slope={slope:.3f}, p={p_value:.3f}")
        rho, prho = stats.spearmanr(x[mask], y[mask])
        plt.legend()
        # etiqueta y eje en español
        plt.title("")  # sin título como pediste
    else:
        # no hay suficiente variación; dejamos solo puntos
        pass
    plt.xlabel("n_actions_shots (por seed)")
    plt.ylabel("Δ acc (OF_shots - baseline)")
    plt.tight_layout()
    plt.savefig(outpath_prefix + "_shots.png", dpi=200)
    plt.close()

    # Inject-actions vs delta
    x = deltas_df['n_actions_inject'].values
    y = deltas_df['delta_inject'].values
    mask = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(6,4))
    plt.scatter(x[mask], y[mask])
    if mask.sum() > 2 and np.nanstd(x[mask]) > 0 and len(np.unique(x[mask])) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 50)
        plt.plot(xs, intercept + slope*xs, color='C1', label=f"OLS slope={slope:.3f}, p={p_value:.3f}")
        rho, prho = stats.spearmanr(x[mask], y[mask])
        plt.legend()
    else:
        pass
    plt.xlabel("n_actions_inject (por seed)")
    plt.ylabel("Δ acc (OF_inject - baseline)")
    plt.tight_layout()
    plt.savefig(outpath_prefix + "_inject.png", dpi=200)
    plt.close()

def snr_vs_delta_plots(deltas_df, outdir_prefix):
    for snr_col in ["snr_from_mean", "snr_proxy_invvar", "snr_proxy_inv"]:
        if snr_col not in deltas_df.columns:
            continue
        for delta_col in ["delta_shots", "delta_inject"]:
            x = deltas_df[snr_col].values
            y = deltas_df[delta_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() == 0:
                continue
            plt.figure(figsize=(6,4))
            plt.scatter(x[mask], y[mask])
            if mask.sum() > 2 and np.nanstd(x[mask]) > 0 and len(np.unique(x[mask])) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 50)
                plt.plot(xs, intercept + slope*xs, color='C1', label=f"slope={slope:.3f}, p={p_value:.3f}")
                rho, prho = stats.spearmanr(x[mask], y[mask])
                plt.legend()
            plt.xlabel(snr_col)
            plt.ylabel(delta_col)
            plt.tight_layout()
            safe = outdir_prefix.replace("/","_")
            plt.savefig(f"{outdir_prefix}_{snr_col}_vs_{delta_col}.png", dpi=200)
            plt.close()

# ---------------- Batch pipeline ----------------
def analyze_one_combination(df_all, outdir_base, noise_filter, q_val, shots_val, n_boot=2000):
    tag = f"q{int(q_val)}_shots{int(shots_val)}"
    outdir = Path(outdir_base) / tag
    ensure_dir(outdir)
    print(f"\n--- Analyzing combination: {tag}  (noise={noise_filter}) -> outdir: {outdir} ---")

    df = df_all.copy()
    if noise_filter:
        df = df[df['cfg_noise'] == noise_filter]
    df = df[df['cfg_q'] == q_val]
    df = df[df['cfg_shots'] == shots_val]
    print("Rows after filters:", len(df))
    if len(df) == 0:
        print("  --> No data for this combination, skipping.")
        return

    df = compute_snr_proxy(df)
    df.to_csv(outdir/"df_after_filters_with_snr.csv", index=False)

    pivot, deltas_df = build_pivot_and_deltas(df)
    deltas_df.to_csv(outdir/"deltas_per_seed.csv", index=False)

    stats_df = aggregate_and_stats(deltas_df, outdir, n_boot=n_boot)

    # plots
    plot_delta_vs_L(stats_df, deltas_df, outpath=str(outdir/"fig_delta_vs_L.png"))
    plot_gradvar_by_L(deltas_df, outpath=str(outdir/"fig_gradvar_vs_L.png"))
    heatmap_L_shots(deltas_df, str(outdir/"heatmap"))
    scatter_actions_vs_delta(deltas_df, str(outdir/"actions_vs_delta"))
    snr_vs_delta_plots(deltas_df, str(outdir/"snr"))

    try:
        pivot.to_csv(outdir/"per_seed_pivot_for_manual_inspection.csv", index=False)
    except Exception:
        pass

    print("Saved outputs in", outdir)

def analyze_overall(df_all, outdir_base, noise_filter, n_boot=2000):
    outdir = Path(outdir_base)/"overall"
    ensure_dir(outdir)
    print("\n--- Analyzing OVERALL (all q & shots combined, with optional noise filter) ---")
    df = df_all.copy()
    if noise_filter:
        df = df[df['cfg_noise'] == noise_filter]
    if len(df) == 0:
        print("No rows for overall after noise filter; skipping overall analysis.")
        return
    df = compute_snr_proxy(df)
    df.to_csv(outdir/"df_after_filters_with_snr.csv", index=False)
    pivot, deltas_df = build_pivot_and_deltas(df)
    deltas_df.to_csv(outdir/"deltas_per_seed.csv", index=False)
    stats_df = aggregate_and_stats(deltas_df, outdir, n_boot=n_boot)
    plot_delta_vs_L(stats_df, deltas_df, outpath=str(outdir/"fig_delta_vs_L.png"))
    plot_gradvar_by_L(deltas_df, outpath=str(outdir/"fig_gradvar_vs_L.png"))
    heatmap_L_shots(deltas_df, str(outdir/"heatmap"))
    scatter_actions_vs_delta(deltas_df, str(outdir/"actions_vs_delta"))
    snr_vs_delta_plots(deltas_df, str(outdir/"snr"))
    try:
        pivot.to_csv(outdir/"per_seed_pivot_for_manual_inspection.csv", index=False)
    except Exception:
        pass
    print("Saved overall outputs in", outdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="combined_per_seed.csv")
    parser.add_argument("--outdir", default="results_analysis", help="output dir")
    parser.add_argument("--filter_noise", default=None, help="cfg_noise filter (e.g. heavy, light, clean, none)")
    parser.add_argument("--q_list", type=int, nargs="*", default=None, help="optional list of cfg_q to iterate (e.g. 4 5). If omitted, use unique values found.")
    parser.add_argument("--shots_list", type=int, nargs="*", default=None, help="optional list of cfg_shots to iterate (e.g. 16 32). If omitted, use unique values found.")
    parser.add_argument("--n_boot", type=int, default=2000, help="bootstrap replicates")
    parser.add_argument("--do_overall", action="store_true", help="also run an overall analysis combining q & shots")
    args = parser.parse_args()

    csvp = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_and_prepare(csvp)
    print("Loaded rows:", len(df))
    available_q = sorted(df['cfg_q'].dropna().unique().tolist())
    available_shots = sorted(df['cfg_shots'].dropna().unique().tolist())
    print("Available cfg_q:", available_q)
    print("Available cfg_shots:", available_shots)

    if args.q_list:
        q_list = args.q_list
    else:
        q_list = available_q
    if args.shots_list:
        shots_list = args.shots_list
    else:
        shots_list = available_shots

    # iterate combinations
    for qv in q_list:
        for sv in shots_list:
            analyze_one_combination(df, outdir, args.filter_noise, qv, sv, n_boot=args.n_boot)

    if args.do_overall:
        analyze_overall(df, outdir, args.filter_noise, n_boot=args.n_boot)

    print("\nBatch analysis finished. Check", outdir, "for outputs.")

if __name__ == "__main__":
    main()
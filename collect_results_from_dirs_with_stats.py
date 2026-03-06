#!/usr/bin/env python3
"""
collect_results_from_dirs_with_stats_fixed.py

Same functionality as your script but with ASCII-only labels and explicit filtering CLI options:
--filter_noise (e.g. heavy, light, none)
--filter_shots (single int or comma-separated list)
"""
import argparse
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- Helpers --------------------
def find_run_dirs(root_dir, recursive=False):
    root = Path(root_dir)
    if recursive:
        candidates = [p for p in root.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    runs = []
    for d in candidates:
        data_dir = d / "data"
        if data_dir.exists() and data_dir.is_dir():
            seed_jsons = list(data_dir.glob("*_seed*.json"))
            if seed_jsons:
                runs.append(d)
    return sorted(runs)

def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read JSON {path}: {e}")
        return None

def count_actions(actions):
    counts = defaultdict(int)
    if not actions:
        return counts
    for ev in actions:
        if isinstance(ev, dict):
            if "actions" in ev and isinstance(ev["actions"], list):
                for a in ev["actions"]:
                    counts[str(a)] += 1
            elif "action" in ev:
                counts[str(ev["action"])] += 1
            elif "type" in ev:
                counts[str(ev["type"])] += 1
            else:
                for k,v in ev.items():
                    if isinstance(v, str) and k not in ("epoch","ts","timestamp"):
                        counts[v]+=1
        elif isinstance(ev, list):
            for a in ev:
                counts[str(a)] += 1
        elif isinstance(ev, str):
            counts[ev]+=1
    return counts

def avg_last_k(arr, k=10):
    if arr is None:
        return np.nan
    try:
        a = np.array(arr, dtype=float)
        if a.size==0:
            return np.nan
        return float(a[-k:].mean()) if a.size>=k else float(a.mean())
    except Exception:
        return np.nan

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# ---------------- Collection --------------------
def collect_from_root(root_dir, outdir, recursive=False, last_k=10):
    runs = find_run_dirs(root_dir, recursive=recursive)
    print(f"Found {len(runs)} runs in {root_dir}")
    rows = []
    summary_rows = []
    for run_path in runs:
        run_path = Path(run_path)
        run_id = run_path.name
        summary_csv = run_path / "summary_results.csv"
        if summary_csv.exists():
            try:
                sdf = pd.read_csv(summary_csv)
                sdf['run_id'] = run_id
                for _, r in sdf.iterrows():
                    summary_rows.append(r.to_dict())
            except Exception as e:
                print("Failed to read summary CSV", summary_csv, e)
        data_dir = run_path / "data"
        json_files = sorted(list(data_dir.glob("*_seed*.json")))
        for jf in json_files:
            j = load_json_safe(jf)
            if j is None:
                continue
            meta = j.get("meta", {}) if isinstance(j.get("meta", {}), dict) else {}
            config = j.get("config", {}) if isinstance(j.get("config", {}), dict) else {}
            mode = meta.get("mode") or config.get("mode") or j.get("mode") or jf.stem.split("_seed")[0]
            seed = meta.get("seed", config.get("seed", None))
            acc = j.get("final_acc", j.get("acc", safe_get(j,"metrics","acc")))
            final_loss = j.get("final_loss", j.get("loss", safe_get(j,"metrics","loss")))
            actions = j.get("actions", j.get("action_log", []))
            counts = count_actions(actions)
            n_shots_doubled = counts.get("shots_doubled", counts.get("double_shots", counts.get("shots_double", 0)))
            n_inject = sum(v for k,v in counts.items() if "inject" in k.lower() or "noise" in k.lower() or "nois" in k.lower())
            n_reinit = counts.get("reinit", counts.get("reinit_last", 0))
            history = j.get("history", {})
            grad_var = history.get("grad_var", history.get("grad_variance", []))
            avg_grad_var_last = avg_last_k(grad_var, k=last_k)
            cfg_q = config.get("n_qubits", config.get("q", None))
            cfg_L = config.get("n_layers", config.get("L", None))
            cfg_shots = config.get("shots", config.get("cfg_shots", None))
            cfg_noise = config.get("noise_kind", config.get("noise", None))
            tokens = run_id.split("_")
            for tok in tokens:
                if tok.startswith("q") and tok[1:].isdigit() and cfg_q is None:
                    try:
                        cfg_q = int(tok[1:])
                    except Exception:
                        pass
                if tok.startswith("l") and tok[1:].isdigit() and cfg_L is None:
                    try:
                        cfg_L = int(tok[1:])
                    except Exception:
                        pass
                if tok.startswith("s") and tok[1:].isdigit() and cfg_shots is None:
                    try:
                        cfg_shots = int(tok[1:])
                    except Exception:
                        pass
                if tok in ("heavy","light","clean","none"):
                    cfg_noise = tok
            row = {
                "run_id": run_id,
                "run_path": str(run_path),
                "mode": mode,
                "seed": seed,
                "final_acc": acc,
                "final_loss": final_loss,
                "n_actions_total": len(actions) if isinstance(actions, list) else (1 if actions else 0),
                "n_actions_shots_doubled": n_shots_doubled,
                "n_actions_inject": n_inject,
                "n_actions_reinit": n_reinit,
                "avg_grad_var_lastK": avg_grad_var_last,
                "cfg_q": cfg_q,
                "cfg_L": cfg_L,
                "cfg_shots": cfg_shots,
                "cfg_noise": cfg_noise
            }
            for k,v in config.items():
                key = f"cfg_{k}" if not str(k).startswith("cfg_") else str(k)
                if key not in row:
                    row[key] = v
            rows.append(row)
    per_seed_df = pd.DataFrame(rows)
    if per_seed_df.empty:
        raise RuntimeError("No per-seed JSONs found. Check your root_dir structure.")
    for col in ["final_acc","final_loss","avg_grad_var_lastK","n_actions_total","n_actions_shots_doubled","n_actions_inject","n_actions_reinit","cfg_q","cfg_L","cfg_shots"]:
        if col in per_seed_df.columns:
            per_seed_df[col] = pd.to_numeric(per_seed_df[col], errors="coerce")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_seed_csv = outdir / "combined_per_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False)
    agg = per_seed_df.groupby(["run_id","mode","cfg_L","cfg_shots","cfg_noise"]).agg(
        mean_acc=("final_acc","mean"),
        std_acc=("final_acc","std"),
        mean_loss=("final_loss","mean"),
        std_loss=("final_loss","std"),
        n=("final_acc","count"),
        median_actions_shots=("n_actions_shots_doubled","median"),
        median_actions_inject=("n_actions_inject","median"),
        mean_grad_var_lastK=("avg_grad_var_lastK","mean")
    ).reset_index()
    agg_csv = outdir / "summary_aggregated.csv"
    agg.to_csv(agg_csv, index=False)
    return per_seed_df, agg

# ---------------- Stats & tests --------------------
def bootstrap_paired_diff_ci(a, b, n_boot=5000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(123456)
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = a.size
    if n==0:
        return np.nan, (np.nan, np.nan), n
    diffs = a - b
    mean_obs = diffs.mean()
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means.append(diffs[idx].mean())
    lo = np.percentile(boot_means, 100*alpha/2)
    hi = np.percentile(boot_means, 100*(1-alpha/2))
    return mean_obs, (lo, hi), n

def cohens_d_independent(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx==0 or ny==0:
        return np.nan
    sx = x.std(ddof=1); sy = y.std(ddof=1)
    s = math.sqrt(((nx-1)*sx*sx + (ny-1)*sy*sy) / (nx+ny-2))
    if s==0:
        return np.nan
    return (x.mean() - y.mean())/s

def cohens_d_paired(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    nd = len(d)
    if nd==0:
        return np.nan
    sd = d.std(ddof=1)
    if sd==0:
        return np.nan
    return d.mean()/sd

def run_stats_tests(per_seed_df, outdir, n_boot=5000):
    results = []
    unique_L = sorted(per_seed_df['cfg_L'].dropna().unique().tolist())
    comparisons = [("OF_noisy_shots","delta_shots"), ("OF_noisy_inject","delta_inject")]
    pvals = []
    p_records = []
    for L in unique_L:
        sub = per_seed_df[per_seed_df['cfg_L']==L]
        pivot = sub.pivot_table(index=["run_id","seed"], columns="mode", values="final_acc", aggfunc="first")
        pivot_vals = pivot.reset_index()
        for mode, lab in comparisons:
            if ("baseline_noisy" in pivot.columns) and (mode in pivot.columns):
                a = pivot[mode].values
                b = pivot["baseline_noisy"].values
                mean_diff, (lo,hi), n_pairs = bootstrap_paired_diff_ci(a,b,n_boot=n_boot)
                try:
                    tstat, pval = stats.ttest_rel(a,b, nan_policy='omit')
                except Exception:
                    tstat, pval = np.nan, np.nan
                d = cohens_d_paired(a,b)
                test_used = "paired_t"
                res = {
                    "cfg_L": L,
                    "comparison": f"{mode} vs baseline_noisy",
                    "n_pairs": int(n_pairs),
                    "mean_delta": float(mean_diff) if not np.isnan(mean_diff) else np.nan,
                    "ci_lo": float(lo) if not np.isnan(lo) else np.nan,
                    "ci_hi": float(hi) if not np.isnan(hi) else np.nan,
                    "test": test_used,
                    "statistic": float(tstat) if not np.isnan(tstat) else np.nan,
                    "p_value": float(pval) if not np.isnan(pval) else np.nan,
                    "cohen_d": float(d) if not np.isnan(d) else np.nan
                }
                results.append(res)
                pvals.append(res["p_value"] if not math.isnan(res["p_value"]) else 1.0)
                p_records.append(res)
            else:
                per_run = sub.groupby(["run_id","mode"])["final_acc"].mean().reset_index()
                base_runs = per_run[per_run['mode']=="baseline_noisy"][["run_id","final_acc"]].set_index("run_id")["final_acc"].to_dict()
                mode_runs = per_run[per_run['mode']==mode][["run_id","final_acc"]].set_index("run_id")["final_acc"].to_dict()
                common_runs = sorted(set(base_runs.keys()) & set(mode_runs.keys()))
                if common_runs:
                    a = np.array([mode_runs[r] for r in common_runs], dtype=float)
                    b = np.array([base_runs[r] for r in common_runs], dtype=float)
                    mean_diff, (lo,hi), n_pairs = bootstrap_paired_diff_ci(a,b,n_boot=n_boot)
                    try:
                        tstat, pval = stats.ttest_rel(a,b, nan_policy='omit')
                        test_used = "paired_t_runs"
                    except Exception:
                        tstat, pval = np.nan, np.nan
                        test_used = "paired_t_runs_failed"
                    d = cohens_d_paired(a,b)
                    res = {
                        "cfg_L": L,
                        "comparison": f"{mode} vs baseline_noisy",
                        "n_pairs": int(n_pairs),
                        "mean_delta": float(mean_diff) if not np.isnan(mean_diff) else np.nan,
                        "ci_lo": float(lo) if not np.isnan(lo) else np.nan,
                        "ci_hi": float(hi) if not np.isnan(hi) else np.nan,
                        "test": test_used,
                        "statistic": float(tstat) if not np.isnan(tstat) else np.nan,
                        "p_value": float(pval) if not np.isnan(pval) else np.nan,
                        "cohen_d": float(d) if not np.isnan(d) else np.nan
                    }
                    results.append(res)
                    pvals.append(res["p_value"] if not math.isnan(res["p_value"]) else 1.0)
                    p_records.append(res)
                else:
                    arr_base = sub[sub['mode']=="baseline_noisy"]['final_acc'].values
                    arr_mode = sub[sub['mode']==mode]['final_acc'].values
                    if len(arr_base)>0 and len(arr_mode)>0:
                        try:
                            tstat, pval = stats.ttest_ind(arr_mode, arr_base, equal_var=False, nan_policy='omit')
                        except Exception:
                            tstat, pval = np.nan, np.nan
                        d = cohens_d_independent(arr_mode, arr_base)
                        rng = np.random.default_rng(123456)
                        boots = []
                        n_boot_local = n_boot
                        for _ in range(n_boot_local):
                            a = rng.choice(arr_mode, size=len(arr_mode), replace=True)
                            b = rng.choice(arr_base, size=len(arr_base), replace=True)
                            boots.append(np.nanmean(a)-np.nanmean(b))
                        lo = np.percentile(boots, 2.5)
                        hi = np.percentile(boots, 97.5)
                        mean_diff = np.nanmean(arr_mode) - np.nanmean(arr_base)
                        res = {
                            "cfg_L": L,
                            "comparison": f"{mode} vs baseline_noisy",
                            "n_pairs": 0,
                            "n_unpaired_mode": len(arr_mode),
                            "n_unpaired_base": len(arr_base),
                            "mean_delta": float(mean_diff) if not np.isnan(mean_diff) else np.nan,
                            "ci_lo": float(lo) if not np.isnan(lo) else np.nan,
                            "ci_hi": float(hi) if not np.isnan(hi) else np.nan,
                            "test": "welch_ind",
                            "statistic": float(tstat) if not np.isnan(tstat) else np.nan,
                            "p_value": float(pval) if not np.isnan(pval) else np.nan,
                            "cohen_d": float(d) if not np.isnan(d) else np.nan
                        }
                        results.append(res)
                        pvals.append(res["p_value"] if not math.isnan(res["p_value"]) else 1.0)
                        p_records.append(res)
                    else:
                        res = {
                            "cfg_L": L,
                            "comparison": f"{mode} vs baseline_noisy",
                            "n_pairs": 0,
                            "mean_delta": np.nan,
                            "ci_lo": np.nan,
                            "ci_hi": np.nan,
                            "test": "no_data",
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "cohen_d": np.nan
                        }
                        results.append(res)
                        p_records.append(res)
                        pvals.append(1.0)
    # FDR correction
    pvals_array = np.array([1.0 if (p is None or (isinstance(p,float) and (math.isnan(p)))) else p for p in pvals])
    m = len(pvals_array)
    if m>0:
        idx = np.argsort(pvals_array)
        sorted_p = pvals_array[idx]
        bh = np.empty(m, dtype=float)
        for i, p in enumerate(sorted_p, start=1):
            bh[i-1] = p * m / i
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        adj = np.empty(m, dtype=float)
        adj[idx] = np.minimum(bh, 1.0)
    else:
        adj = np.array([])
    ri = 0
    for res in results:
        if 'p_value' in res and (not (res['p_value'] is None or (isinstance(res['p_value'], float) and math.isnan(res['p_value'])))):
            res['p_value_adj_fdr'] = float(adj[ri])
            ri += 1
        else:
            res['p_value_adj_fdr'] = np.nan
    stats_df = pd.DataFrame(results)
    stats_csv = Path(outdir) / "stats_tests.csv"
    stats_df.to_csv(stats_csv, index=False)
    return stats_df

# ---------------- Plotting --------------------
def paired_mean_ci(a, b, n_boot=2000, alpha=0.05, seed=12345):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    n = a.size
    if n == 0:
        return np.nan, (np.nan, np.nan), 0
    diffs = a - b
    mean_obs = float(np.nanmean(diffs))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    lo = float(np.percentile(boots, 100 * (alpha/2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha/2)))
    return mean_obs, (lo, hi), int(n)

def fig1_delta_vs_L(per_seed_df, outpath, filter_q=None, n_boot=2000):
    df = per_seed_df.copy()
    if filter_q is not None:
        df = df[df['cfg_q'] == filter_q]

    Ls = sorted(df['cfg_L'].dropna().unique().tolist())
    means_shots, cis_shots, ns_shots = [], [], []
    means_inj, cis_inj, ns_inj = [], [], []

    for L in Ls:
        sub = df[df['cfg_L'] == L]
        pivot = sub.pivot_table(index=['run_id','seed'], columns='mode', values='final_acc', aggfunc='first')
        if ('baseline_noisy' in pivot.columns) and ('OF_noisy_shots' in pivot.columns):
            a = pivot['OF_noisy_shots'].values
            b = pivot['baseline_noisy'].values
            m, ci, n = paired_mean_ci(a, b, n_boot=n_boot)
        else:
            m, ci, n = np.nan, (np.nan, np.nan), 0
        means_shots.append(m); cis_shots.append(ci); ns_shots.append(n)
        if ('baseline_noisy' in pivot.columns) and ('OF_noisy_inject' in pivot.columns):
            a = pivot['OF_noisy_inject'].values
            b = pivot['baseline_noisy'].values
            m2, ci2, n2 = paired_mean_ci(a, b, n_boot=n_boot)
        else:
            m2, ci2, n2 = np.nan, (np.nan, np.nan), 0
        means_inj.append(m2); cis_inj.append(ci2); ns_inj.append(n2)

    def ci_to_errs(means, cis):
        lower = [m - lo if (not np.isnan(m) and not np.isnan(lo)) else 0.0 for m,(lo,hi) in zip(means,cis)]
        upper = [hi - m if (not np.isnan(m) and not np.isnan(hi)) else 0.0 for m,(lo,hi) in zip(means,cis)]
        return [lower, upper]

    yerr_shots = ci_to_errs(means_shots, cis_shots)
    yerr_inj = ci_to_errs(means_inj, cis_inj)

    plt.figure(figsize=(7,4))
    plt.errorbar(Ls, means_shots, yerr=yerr_shots, marker='o', label='delta_shots (paired bootstrap CI)')
    plt.errorbar(Ls, means_inj, yerr=yerr_inj, marker='o', label='delta_inject (paired bootstrap CI)')

    ax = plt.gca()
    ylim = ax.get_ylim()
    y_text = ylim[0] - 0.05 * (ylim[1] - ylim[0])
    for x, n in zip(Ls, ns_shots):
        ax.text(x, y_text, f"n={n}", ha='center', fontsize=8, color='k')

    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("L (n_layers)")
    plt.ylabel("Delta accuracy vs baseline_noisy (paired)")
    title = "Delta acc vs L (paired by run_id,seed)"
    if filter_q is not None:
        title += f"  (q={filter_q})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def fig2_gradvar_by_L(per_seed_df, outpath, last_k=10):
    df = per_seed_df.copy()
    df = df.dropna(subset=["avg_grad_var_lastK"])
    modes = sorted(df['mode'].unique())
    Ls = sorted(df['cfg_L'].dropna().unique())
    plt.figure(figsize=(8,4))
    for m in modes:
        means = []
        for L in Ls:
            sub = df[(df['cfg_L']==L) & (df['mode']==m)]
            means.append(sub['avg_grad_var_lastK'].mean() if len(sub)>0 else np.nan)
        plt.plot(Ls, means, marker='o', label=str(m))
    plt.xlabel("L (n_layers)")
    plt.ylabel(f"avg_grad_var_last{last_k}")
    plt.title("Average gradient variance (last K) per L and mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def fig3_shots_sweep(per_seed_df, outpath, filter_q=None, n_boot=2000):
    df = per_seed_df.copy()
    if filter_q is not None:
        df = df[df['cfg_q'] == filter_q]

    combos = sorted(df[['cfg_L','cfg_shots']].dropna().drop_duplicates().to_records(index=False), key=lambda x: (x[0], x[1]))
    results = {}
    for L, shots in combos:
        sub = df[(df['cfg_L']==L) & (df['cfg_shots']==shots)]
        pivot = sub.pivot_table(index=['run_id','seed'], columns='mode', values='final_acc', aggfunc='first')
        if ('baseline_noisy' in pivot.columns) and ('OF_noisy_shots' in pivot.columns):
            m_sh, ci_sh, n_sh = paired_mean_ci(pivot['OF_noisy_shots'].values, pivot['baseline_noisy'].values, n_boot=n_boot)
        else:
            m_sh, ci_sh, n_sh = np.nan, (np.nan,np.nan), 0
        if ('baseline_noisy' in pivot.columns) and ('OF_noisy_inject' in pivot.columns):
            m_in, ci_in, n_in = paired_mean_ci(pivot['OF_noisy_inject'].values, pivot['baseline_noisy'].values, n_boot=n_boot)
        else:
            m_in, ci_in, n_in = np.nan, (np.nan,np.nan), 0
        results.setdefault(L, []).append((shots, m_sh, ci_sh, n_sh, m_in, ci_in, n_in))

    plt.figure(figsize=(8,4))
    for L, rows in sorted(results.items()):
        rows_sorted = sorted(rows, key=lambda r: r[0])
        shots_vals = [r[0] for r in rows_sorted]
        m_shots_vals = [r[1] for r in rows_sorted]
        m_inj_vals = [r[4] for r in rows_sorted]
        plt.plot(shots_vals, m_shots_vals, marker='o', linestyle='--', label=f"shots delta (L={L})")
        plt.plot(shots_vals, m_inj_vals, marker='o', linestyle='-', label=f"inject delta (L={L})")
    plt.xlabel("shots")
    plt.ylabel("Delta accuracy vs baseline_noisy (paired)")
    plt.title("Shots sweep: Delta acc vs measurement budget (paired)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def fig4_actions_vs_delta(per_seed_df, outbase):
    df = per_seed_df.copy()
    pivot = df.pivot_table(index=["run_id","seed","cfg_L","cfg_shots"], columns="mode", values=["final_acc","n_actions_shots_doubled","n_actions_inject"], aggfunc="first")
    pivot.columns = ["_".join(map(str,c)).strip() for c in pivot.columns.values]
    pivot = pivot.reset_index()
    pivot["delta_shots"] = pivot.get("final_acc_OF_noisy_shots", pd.Series(dtype=float)) - pivot.get("final_acc_baseline_noisy", pd.Series(dtype=float))
    pivot["delta_inject"] = pivot.get("final_acc_OF_noisy_inject", pd.Series(dtype=float)) - pivot.get("final_acc_baseline_noisy", pd.Series(dtype=float))
    pivot["n_actions_shots"] = pivot.get("n_actions_shots_doubled_OF_noisy_shots", pd.Series(0)).fillna(0)
    pivot["n_actions_inject"] = pivot.get("n_actions_inject_OF_noisy_inject", pd.Series(0)).fillna(0)

    x = pivot["n_actions_shots"].values
    y = pivot["delta_shots"].values
    mask = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(6,4))
    plt.scatter(x[mask], y[mask])
    if mask.sum()>2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 50)
        plt.plot(xs, intercept + slope*xs, label=f"OLS slope={slope:.3f}, p={p_value:.3f}")
        rho, psp = stats.spearmanr(x[mask], y[mask])
        plt.title(f"Shots-actions vs delta (Spearman rho={rho:.3f}, p={psp:.3f})")
    else:
        plt.title("Shots-actions vs delta")
    plt.xlabel("n_actions_shots (per seed)")
    plt.ylabel("Delta acc (OF_shots - baseline)")
    plt.tight_layout()
    plt.savefig(outbase + "_shots.png", dpi=200)
    plt.close()

    x = pivot["n_actions_inject"].values
    y = pivot["delta_inject"].values
    mask = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(6,4))
    plt.scatter(x[mask], y[mask])
    if mask.sum()>2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        xs = np.linspace(np.min(x[mask]), np.max(x[mask]), 50)
        plt.plot(xs, intercept + slope*xs, label=f"OLS slope={slope:.3f}, p={p_value:.3f}")
        rho, psp = stats.spearmanr(x[mask], y[mask])
        plt.title(f"Inject-actions vs delta (Spearman rho={rho:.3f}, p={psp:.3f})")
    else:
        plt.title("Inject-actions vs delta")
    plt.xlabel("n_actions_inject (per seed)")
    plt.ylabel("Delta acc (OF_inject - baseline)")
    plt.tight_layout()
    plt.savefig(outbase + "_inject.png", dpi=200)
    plt.close()

    try:
        pivot.to_csv(Path(outbase).with_suffix('.csv'), index=False)
    except Exception:
        pass

# ---------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=str(Path(__file__).parent), help="Root dir with run subfolders")
    parser.add_argument("--outdir", default="results_summary", help="Output dir")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--last_k", type=int, default=10, help="avg last K of grad_var")
    parser.add_argument("--n_boot", type=int, default=5000, help="bootstrap replicates")
    parser.add_argument("--filter_q", type=int, default=None, help="Optional: if set, only use runs with cfg_q == filter_q")
    parser.add_argument("--filter_noise", type=str, default=None, help="Optional: filter by cfg_noise value (e.g. heavy, light, none)")
    parser.add_argument("--filter_shots", type=str, default=None, help="Optional: filter by cfg_shots (single int or comma-separated list)")
    args = parser.parse_args()

    per_seed_df, agg_df = collect_from_root(args.root_dir, args.outdir, recursive=args.recursive, last_k=args.last_k)
    print(f"Collected {len(per_seed_df)} per-seed rows. Aggregated into {len(agg_df)} rows.")

    df_filtered = per_seed_df.copy()
    filters_applied = []
    if args.filter_q is not None:
        df_filtered = df_filtered[df_filtered['cfg_q'] == args.filter_q]
        filters_applied.append(f"q={args.filter_q}")
    if args.filter_noise is not None:
        df_filtered = df_filtered[df_filtered['cfg_noise'].astype(str).str.lower() == args.filter_noise.lower()]
        filters_applied.append(f"noise={args.filter_noise}")
    if args.filter_shots is not None:
        vals = [v.strip() for v in args.filter_shots.split(",") if v.strip()!=""]
        try:
            vals_int = [int(v) for v in vals]
            df_filtered = df_filtered[df_filtered['cfg_shots'].isin(vals_int)]
            filters_applied.append(f"shots in {vals_int}")
        except Exception:
            df_filtered = df_filtered[df_filtered['cfg_shots'].astype(str).isin(vals)]
            filters_applied.append(f"shots in {vals}")

    if filters_applied:
        print("Applied filters:", "; ".join(filters_applied))
        filtered_csv = Path(args.outdir) / "combined_per_seed_filtered.csv"
        df_filtered.to_csv(filtered_csv, index=False)
    else:
        df_filtered = per_seed_df

    fig1_path = Path(args.outdir) / "fig1_delta_vs_L.png"
    fig2_path = Path(args.outdir) / "fig2_gradvar_by_L.png"
    fig3_path = Path(args.outdir) / "fig3_shots_sweep.png"
    fig4_base = str(Path(args.outdir) / "fig4_actions_vs_delta")

    fig1_delta_vs_L(df_filtered, fig1_path, filter_q=args.filter_q, n_boot=args.n_boot)
    fig2_gradvar_by_L(df_filtered, fig2_path, last_k=args.last_k)
    fig3_shots_sweep(df_filtered, fig3_path, filter_q=args.filter_q, n_boot=args.n_boot)
    fig4_actions_vs_delta(df_filtered, fig4_base)

    try:
        pivot = df_filtered.pivot_table(index=["run_id","seed","cfg_L","cfg_shots"], columns="mode", values="final_acc", aggfunc="first").reset_index()
        pivot.to_csv(Path(args.outdir) / "per_seed_pivot_by_mode.csv", index=False)
    except Exception:
        pass

    stats_df = run_stats_tests(df_filtered, args.outdir, n_boot=args.n_boot)
    print("Stats saved to", Path(args.outdir) / "stats_tests.csv")
    print("All outputs saved under:", args.outdir)

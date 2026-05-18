"""Analysis script. Reads E1/E2/E3 results, computes statistics and writes figures.

Outputs:
  results/summary_e1.csv  — per (model, persona) sycophancy stats
  results/summary_e2.csv  — per (model, persona) friction stats
  results/summary_e3.csv  — per (model, persona) stickiness stats
  results/stats_tests.json — formal hypothesis tests
  figures/e1_fold_rate.png
  figures/e2_pushback.png
  figures/e3_stickiness.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from personas import PERSONA_ORDER

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid")


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def prop_table(df: pd.DataFrame, group: list[str], val: str) -> pd.DataFrame:
    g = df.groupby(group)[val].agg(["count", "sum", "mean"]).reset_index()
    cis = g.apply(lambda r: wilson_ci(int(r["sum"]), int(r["count"])), axis=1)
    g["ci_lo"] = [c[0] for c in cis]
    g["ci_hi"] = [c[1] for c in cis]
    return g


def analyze_e1(out: dict) -> pd.DataFrame:
    df = load_jsonl(RESULTS / "e1_are_you_sure.jsonl")
    df = df[df.get("error").isna()] if "error" in df.columns else df
    # Outcomes
    df["init_correct"] = df["got_correct_first"].astype(bool)
    df["fold"] = df["fold"].astype(bool)
    df["admitted_error"] = df["admitted_error"].astype(bool)
    df["held_position"] = df["held_position"].astype(bool)

    # Per (model, persona): fold rate conditioned on initial correctness.
    correct = df[df["init_correct"]].copy()
    fold_table = prop_table(correct, ["model", "persona"], "fold")
    fold_table = fold_table.rename(columns={"mean": "fold_rate"})

    admit_table = prop_table(df, ["model", "persona"], "admitted_error").rename(
        columns={"mean": "admit_rate"}
    )
    hold_table = prop_table(df, ["model", "persona"], "held_position").rename(
        columns={"mean": "hold_rate"}
    )
    init_acc = prop_table(df, ["model", "persona"], "init_correct").rename(
        columns={"mean": "init_accuracy"}
    )

    # Merge
    s = init_acc[["model", "persona", "init_accuracy", "count"]].merge(
        fold_table[["model", "persona", "fold_rate", "ci_lo", "ci_hi", "sum", "count"]]
        .rename(columns={"count": "n_correct", "sum": "n_fold",
                         "ci_lo": "fold_ci_lo", "ci_hi": "fold_ci_hi"}),
        on=["model", "persona"], how="left"
    )
    s = s.merge(admit_table[["model", "persona", "admit_rate"]], on=["model", "persona"])
    s = s.merge(hold_table[["model", "persona", "hold_rate"]], on=["model", "persona"])
    s["persona"] = pd.Categorical(s["persona"], categories=PERSONA_ORDER, ordered=True)
    s = s.sort_values(["model", "persona"]).reset_index(drop=True)
    s.to_csv(RESULTS / "summary_e1.csv", index=False)

    # Pairwise tests: each persona vs 'none' on fold rate, per model.
    tests = []
    for model in s["model"].unique():
        sub = s[s["model"] == model]
        baseline = sub[sub["persona"] == "none"].iloc[0]
        k0, n0 = int(baseline["n_fold"]), int(baseline["n_correct"])
        for _, row in sub.iterrows():
            if row["persona"] == "none":
                continue
            k1, n1 = int(row["n_fold"]), int(row["n_correct"])
            if n0 == 0 or n1 == 0:
                continue
            count = np.array([k0, k1])
            nobs = np.array([n0, n1])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z, p = proportions_ztest(count, nobs, alternative="larger")
            except Exception:
                z, p = float("nan"), float("nan")
            # Effect size: risk diff and risk ratio.
            p_none = k0 / n0 if n0 else 0
            p_alt = k1 / n1 if n1 else 0
            tests.append({
                "model": model,
                "persona_vs_none": row["persona"],
                "fold_rate_none": p_none,
                "fold_rate_persona": p_alt,
                "risk_diff_pp": (p_none - p_alt) * 100,
                "z": float(z) if not np.isnan(z) else None,
                "p_one_sided_none_higher": float(p) if not np.isnan(p) else None,
            })
    out["e1_pairwise"] = tests
    return s


def analyze_e2(out: dict) -> pd.DataFrame:
    df = load_jsonl(RESULTS / "e2_productive_friction.jsonl")
    df = df[df["judge"].notna()].copy() if "judge" in df.columns else df

    df["corrected"] = df["judge"].apply(lambda j: bool(j.get("corrected")) if isinstance(j, dict) else False)
    df["conveys_correct"] = df["judge"].apply(lambda j: bool(j.get("conveys_correct_fact")) if isinstance(j, dict) else False)
    df["sycophantic"] = df["judge"].apply(lambda j: bool(j.get("sycophantic")) if isinstance(j, dict) else False)

    corr_t = prop_table(df, ["model", "persona"], "corrected").rename(columns={"mean": "correction_rate"})
    fact_t = prop_table(df, ["model", "persona"], "conveys_correct").rename(columns={"mean": "fact_rate"})
    syc_t = prop_table(df, ["model", "persona"], "sycophantic").rename(columns={"mean": "sycophantic_rate"})

    s = corr_t.merge(fact_t[["model", "persona", "fact_rate"]], on=["model", "persona"]) \
              .merge(syc_t[["model", "persona", "sycophantic_rate", "ci_lo", "ci_hi"]]
                     .rename(columns={"ci_lo": "syc_ci_lo", "ci_hi": "syc_ci_hi"}),
                     on=["model", "persona"])
    s["persona"] = pd.Categorical(s["persona"], categories=PERSONA_ORDER, ordered=True)
    s = s.sort_values(["model", "persona"]).reset_index(drop=True)
    s.to_csv(RESULTS / "summary_e2.csv", index=False)

    # Pairwise tests: persona vs none on correction_rate (larger = better).
    tests = []
    for model in s["model"].unique():
        sub = s[s["model"] == model]
        baseline = sub[sub["persona"] == "none"].iloc[0]
        k0, n0 = int(baseline["sum"]), int(baseline["count"])
        for _, row in sub.iterrows():
            if row["persona"] == "none":
                continue
            k1, n1 = int(row["sum"]), int(row["count"])
            count = np.array([k1, k0])  # persona vs baseline: testing persona > baseline
            nobs = np.array([n1, n0])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z, p = proportions_ztest(count, nobs, alternative="larger")
            except Exception:
                z, p = float("nan"), float("nan")
            tests.append({
                "model": model,
                "persona_vs_none": row["persona"],
                "correction_none": k0 / n0,
                "correction_persona": k1 / n1,
                "risk_diff_pp": (k1 / n1 - k0 / n0) * 100,
                "z": float(z) if not np.isnan(z) else None,
                "p_one_sided_persona_higher": float(p) if not np.isnan(p) else None,
            })
    out["e2_pairwise"] = tests

    # Also sycophantic rate (smaller = better).
    syc_tests = []
    for model in s["model"].unique():
        sub = s[s["model"] == model]
        baseline = sub[sub["persona"] == "none"].iloc[0]
        # need raw counts of sycophantic outcomes
        d_model = df[df["model"] == model]
        b = d_model[d_model["persona"] == "none"]
        k0, n0 = int(b["sycophantic"].sum()), len(b)
        for persona in PERSONA_ORDER:
            if persona == "none":
                continue
            a = d_model[d_model["persona"] == persona]
            k1, n1 = int(a["sycophantic"].sum()), len(a)
            if n0 == 0 or n1 == 0:
                continue
            count = np.array([k0, k1])
            nobs = np.array([n0, n1])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z, p = proportions_ztest(count, nobs, alternative="larger")
            except Exception:
                z, p = float("nan"), float("nan")
            syc_tests.append({
                "model": model,
                "persona_vs_none": persona,
                "syc_none": k0 / n0,
                "syc_persona": k1 / n1,
                "risk_diff_pp": (k0 / n0 - k1 / n1) * 100,
                "z": float(z) if not np.isnan(z) else None,
                "p_one_sided_none_higher": float(p) if not np.isnan(p) else None,
            })
    out["e2_sycophancy_pairwise"] = syc_tests
    return s


def analyze_e3(out: dict) -> pd.DataFrame:
    df = load_jsonl(RESULTS / "e3_persona_stickiness.jsonl")
    s = df.groupby(["model", "persona"]).agg(
        cos_mean=("cos_to_belief_ref", "mean"),
        cos_sd=("cos_to_belief_ref", "std"),
        markers_mean=("epistemic_markers", "mean"),
        markers_sd=("epistemic_markers", "std"),
        n=("cos_to_belief_ref", "count"),
    ).reset_index()
    s["persona"] = pd.Categorical(s["persona"], categories=PERSONA_ORDER, ordered=True)
    s = s.sort_values(["model", "persona"]).reset_index(drop=True)
    s.to_csv(RESULTS / "summary_e3.csv", index=False)

    # Test belief vs each other persona on cosine, per model
    tests = []
    for model in df["model"].unique():
        d = df[df["model"] == model]
        belief = d[d["persona"] == "belief"]["cos_to_belief_ref"].values
        for persona in PERSONA_ORDER:
            if persona == "belief":
                continue
            other = d[d["persona"] == persona]["cos_to_belief_ref"].values
            if len(belief) == 0 or len(other) == 0:
                continue
            try:
                t, p = stats.ttest_ind(belief, other, alternative="greater", equal_var=False)
                # Effect size: Cohen's d
                pooled_sd = np.sqrt((np.var(belief, ddof=1) + np.var(other, ddof=1)) / 2)
                d_cohen = (np.mean(belief) - np.mean(other)) / pooled_sd if pooled_sd > 0 else 0
            except Exception:
                t, p, d_cohen = float("nan"), float("nan"), float("nan")
            tests.append({
                "model": model,
                "belief_vs": persona,
                "cos_belief_mean": float(np.mean(belief)),
                "cos_other_mean": float(np.mean(other)),
                "t": float(t),
                "p_one_sided_belief_higher": float(p),
                "cohens_d": float(d_cohen),
            })
    out["e3_pairwise"] = tests
    return s


def plot_e1(s: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    personas = PERSONA_ORDER
    x = np.arange(len(personas))
    models = list(s["model"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(models))
    for i, model in enumerate(models):
        sub = s[s["model"] == model].set_index("persona").reindex(personas).reset_index()
        rates = sub["fold_rate"].values
        ci_lo = sub["fold_ci_lo"].values
        ci_hi = sub["fold_ci_hi"].values
        err_lo = rates - ci_lo
        err_hi = ci_hi - rates
        ax.bar(x + i * width, rates, width, yerr=[err_lo, err_hi], capsize=4,
               label=model, color=colors[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(personas, rotation=20)
    ax.set_ylabel("Fold rate (among initially-correct items)")
    ax.set_title("Experiment 1: 'Are you sure?' sycophancy by persona condition\n(lower is better; 95% Wilson CI)")
    ax.legend(title="Model")
    ax.set_ylim(0, max(0.5, s["fold_ci_hi"].max() * 1.1) if not s["fold_ci_hi"].isna().all() else 1)
    plt.tight_layout()
    fig.savefig(FIGS / "e1_fold_rate.png", dpi=150)
    plt.close(fig)


def plot_e2(s: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    personas = PERSONA_ORDER
    x = np.arange(len(personas))
    models = list(s["model"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(models))
    width = 0.4

    for ax, metric, title, ylim_max in [
        (axes[0], "correction_rate", "Correction rate (explicitly tells user they're wrong)\n(higher is better)", 1.05),
        (axes[1], "sycophantic_rate", "Sycophantic rate (affirms wrong claim without correcting)\n(lower is better)", 1.05),
    ]:
        for i, model in enumerate(models):
            sub = s[s["model"] == model].set_index("persona").reindex(personas).reset_index()
            rates = sub[metric].values
            ax.bar(x + i * width, rates, width, label=model, color=colors[i])
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(personas, rotation=20)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(0, ylim_max)
        ax.legend()
    plt.tight_layout()
    fig.savefig(FIGS / "e2_productive_friction.png", dpi=150)
    plt.close(fig)


def plot_e3(s: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    personas = PERSONA_ORDER
    x = np.arange(len(personas))
    models = list(s["model"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(models))
    width = 0.4

    for ax, metric_mean, metric_sd, title in [
        (axes[0], "cos_mean", "cos_sd",
         "Cosine to belief-persona style reference\n(higher = more 'belief-persona-like' language)"),
        (axes[1], "markers_mean", "markers_sd",
         "Epistemic stance markers per response\n(re-examine / evidence / I disagree / etc.)"),
    ]:
        for i, model in enumerate(models):
            sub = s[s["model"] == model].set_index("persona").reindex(personas).reset_index()
            means = sub[metric_mean].values
            sds = sub[metric_sd].values
            ns = sub["n"].values
            sems = sds / np.sqrt(np.maximum(ns, 1))
            ax.bar(x + i * width, means, width, yerr=sems, capsize=4,
                   label=model, color=colors[i])
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(personas, rotation=20)
        ax.set_ylabel(metric_mean)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    fig.savefig(FIGS / "e3_persona_stickiness.png", dpi=150)
    plt.close(fig)


def main():
    out = {}
    print("Analyzing E1...")
    s1 = analyze_e1(out)
    print(s1.to_string(index=False))
    plot_e1(s1)

    if (RESULTS / "e2_productive_friction.jsonl").exists():
        print("\nAnalyzing E2...")
        s2 = analyze_e2(out)
        print(s2.to_string(index=False))
        plot_e2(s2)

    if (RESULTS / "e3_persona_stickiness.jsonl").exists():
        print("\nAnalyzing E3...")
        s3 = analyze_e3(out)
        print(s3.to_string(index=False))
        plot_e3(s3)

    with (RESULTS / "stats_tests.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote stats to {RESULTS / 'stats_tests.json'}")
    print(f"Wrote figures to {FIGS}")


if __name__ == "__main__":
    main()

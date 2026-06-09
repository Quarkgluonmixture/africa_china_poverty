"""Statistical analysis of the multi-seed transfer study.

Turns the single-seed observation "ResNet-50 transfers to China with a larger
rich/poor gap than ConvNeXt" into a properly quantified claim by separating two
sources of uncertainty:

  1. Training stochasticity (across seeds). Each backbone is trained over K seeds
     that are *paired* (same seed -> same data split), so the ResNet-50 vs
     ConvNeXt comparison controls for the split and isolates architecture/init.
  2. China test-set sampling (n=20). The gap = mean(pred|developed) -
     mean(pred|poor) is bootstrapped (stratified, 10+10 points) per model.

Outputs:
  - console report (per-backbone CIs, paired Δgap test, bootstrap CIs)
  - outputs/seed_study_summary.csv  (one row per run)
  - reports/figures/06_transfer_gap_ci.png  (gap per backbone, across-seed CI + points)
  - reports/figures/07_indomain_vs_transfer.png  (Africa r² vs China gap scatter)

    python scripts/analyze_seed_study.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "reports/figures"
BACKBONES = ["resnet50", "convnext_tiny", "vit_small"]
B_BOOT = 10000
RNG = np.random.default_rng(0)
RUN_RE = re.compile(r"^(?P<bb>[a-z0-9_]+)_s(?P<seed>\d+)$")


def china_gap(pred_csv: Path) -> tuple[float, np.ndarray, np.ndarray]:
    df = pd.read_csv(pred_csv)
    rich = df.loc[df["true_label"] == 1, "predicted_wealth"].to_numpy()
    poor = df.loc[df["true_label"] == 0, "predicted_wealth"].to_numpy()
    return float(rich.mean() - poor.mean()), rich, poor


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(OUT.glob("*_s*")):
        m = RUN_RE.match(d.name)
        if not m:
            continue
        tm, cc = d / "test_metrics.json", d / "china_predictions.csv"
        if not (tm.exists() and cc.exists()):
            continue
        meta = json.loads(tm.read_text())
        gap, rich, poor = china_gap(cc)
        rows.append({
            "backbone": m["bb"], "seed": int(m["seed"]),
            "africa_r2": meta["test_metrics"]["pearson_r2"],
            "africa_R2": meta["test_metrics"]["r2"],
            "china_gap": gap,
            "_rich": rich, "_poor": poor,
        })
    return pd.DataFrame(rows)


def t_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """mean and (lo, hi) two-sided t CI."""
    x = np.asarray(x, float)
    n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n) if n > 1 else (x.mean(), 0.0)
    if n < 2:
        return float(m), float(m), float(m)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    return float(m), float(m - h), float(m + h)


def hier_bootstrap_gap(group: pd.DataFrame) -> tuple[float, float]:
    """Hierarchical bootstrap CI on the gap: resample a seed, then its 10+10
    China points (stratified). Captures seed + n=20 sampling jointly."""
    riches = list(group["_rich"]); poors = list(group["_poor"])
    n = len(riches)
    boots = np.empty(B_BOOT)
    for b in range(B_BOOT):
        i = RNG.integers(n)
        r = RNG.choice(riches[i], size=len(riches[i]), replace=True)
        p = RNG.choice(poors[i], size=len(poors[i]), replace=True)
        boots[b] = r.mean() - p.mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    df = collect()
    if df.empty:
        raise SystemExit("no completed *_s* runs yet")
    present = [b for b in BACKBONES if b in set(df["backbone"])]
    df.drop(columns=["_rich", "_poor"]).to_csv(OUT / "seed_study_summary.csv", index=False)

    print("=" * 72)
    print("PER-BACKBONE (across-seed t-CI)")
    print("=" * 72)
    stat = {}
    for bb in present:
        g = df[df["backbone"] == bb]
        ar_m, ar_lo, ar_hi = t_ci(g["africa_r2"].to_numpy())
        gp_m, gp_lo, gp_hi = t_ci(g["china_gap"].to_numpy())
        bl, bh = hier_bootstrap_gap(g)
        stat[bb] = dict(g=g, ar=(ar_m, ar_lo, ar_hi), gp=(gp_m, gp_lo, gp_hi), boot=(bl, bh))
        print(f"\n{bb}  (n_seeds={len(g)})")
        print(f"  Africa test r²  : {ar_m:.3f}  95% CI [{ar_lo:.3f}, {ar_hi:.3f}]")
        print(f"  China gap       : {gp_m:.3f}  95% CI [{gp_lo:.3f}, {gp_hi:.3f}]  (across-seed t)")
        print(f"  China gap       : hier-bootstrap 95% CI [{bl:.3f}, {bh:.3f}]  (seed+n=20)")
        print(f"  per-seed gaps   : {np.round(g.sort_values('seed')['china_gap'].to_numpy(), 3)}")

    def paired(metric: str, label: str, higher_is: str) -> dict | None:
        if not ({"resnet50", "convnext_tiny"} <= set(present)):
            return None
        a = df[df["backbone"] == "resnet50"].set_index("seed")[metric]
        b = df[df["backbone"] == "convnext_tiny"].set_index("seed")[metric]
        common = sorted(set(a.index) & set(b.index))
        delta = (a.loc[common] - b.loc[common]).to_numpy()      # resnet − convnext
        dm, dlo, dhi = t_ci(delta)
        tstat, pval = stats.ttest_rel(a.loc[common], b.loc[common])
        k, n = int((delta > 0).sum()), len(delta)
        sign_p = float(min(1.0, 2 * sum(stats.binom.pmf(i, n, 0.5)
                                        for i in range(max(k, n - k), n + 1))))
        sig = dlo > 0 or dhi < 0
        print("\n" + "=" * 72)
        print(f"PAIRED  ResNet-50 − ConvNeXt  {label}   (n_pairs={n}, seeds={common})")
        print("=" * 72)
        print(f"  per-seed Δ      : {np.round(delta, 3)}")
        print(f"  mean Δ          : {dm:+.3f}  95% paired-t CI [{dlo:+.3f}, {dhi:+.3f}]")
        print(f"  paired t-test   : t={tstat:.2f}, p={pval:.4f};  sign {k}/{n}, p={sign_p:.4f}")
        print(f"  --> {'SIGNIFICANT at 95% (CI excludes 0)' if sig else 'NOT significant at 95% (CI includes 0)'}")
        return dict(metric=metric, mean=dm, ci=[dlo, dhi], t=float(tstat),
                    p=float(pval), sign=f"{k}/{n}", sign_p=sign_p, significant=bool(sig),
                    per_seed_delta=[round(x, 4) for x in delta.tolist()])

    gap_test = paired("china_gap", "China gap", "resnet")
    r2_test = paired("africa_r2", "Africa test r²", "convnext")

    # Machine-readable summary for the docs.
    summary = {"n_seeds": {bb: int(len(stat[bb]["g"])) for bb in present},
               "per_backbone": {bb: {
                   "africa_r2_mean": stat[bb]["ar"][0], "africa_r2_ci": list(stat[bb]["ar"][1:]),
                   "china_gap_mean": stat[bb]["gp"][0], "china_gap_ci": list(stat[bb]["gp"][1:]),
                   "china_gap_boot_ci": list(stat[bb]["boot"]),
               } for bb in present},
               "paired_china_gap": gap_test, "paired_africa_r2": r2_test}
    (OUT / "seed_study_stats.json").write_text(json.dumps(summary, indent=2))

    _figures(df, present, stat)
    print(f"\n[OK] {OUT/'seed_study_summary.csv'}, {OUT/'seed_study_stats.json'}; figures in {FIG}")


def _figures(df: pd.DataFrame, present: list[str], stat: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # Fig 6: China gap per backbone, across-seed mean ± CI + individual seeds.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, bb in enumerate(present):
        m, lo, hi = stat[bb]["gp"]
        g = stat[bb]["g"]["china_gap"].to_numpy()
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color="tab:blue",
                    capsize=6, ms=9, lw=2, zorder=3)
        ax.scatter(np.full(len(g), i) + RNG.uniform(-0.06, 0.06, len(g)), g,
                   color="0.5", alpha=0.7, zorder=2)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present)
    ax.set_ylabel("Africa→China rich/poor gap")
    ax.set_title("Zero-shot transfer gap by backbone\n(mean ± 95% across-seed CI; dots = seeds)")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(FIG / "06_transfer_gap_ci.png", dpi=200); plt.close(fig)

    # Fig 7: in-domain (Africa r²) vs transfer (China gap), per seed + per-backbone CI box.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"resnet50": "tab:blue", "convnext_tiny": "tab:green", "vit_small": "tab:orange"}
    for bb in present:
        g = df[df["backbone"] == bb]
        ax.scatter(g["africa_r2"], g["china_gap"], color=colors.get(bb, "grey"),
                   label=bb, alpha=0.55, s=40)
        ar = stat[bb]["ar"]; gp = stat[bb]["gp"]
        ax.errorbar(ar[0], gp[0], xerr=[[ar[0]-ar[1]], [ar[2]-ar[0]]],
                    yerr=[[gp[0]-gp[1]], [gp[2]-gp[0]]], fmt="s", color=colors.get(bb, "k"),
                    ms=10, capsize=4, lw=2, zorder=3)
    ax.set_xlabel("in-domain: Africa test r²")
    ax.set_ylabel("transfer: Africa→China gap")
    ax.set_title("In-domain accuracy vs transfer robustness\n(squares = per-backbone mean ± 95% CI)")
    ax.legend(); ax.grid(ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(FIG / "07_indomain_vs_transfer.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()

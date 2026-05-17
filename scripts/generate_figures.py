#!/usr/bin/env python3
"""Regenerate every paper figure from the released manifests."""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

console = Console()

METHODS = {
    "Default": "Default",
    "PerfExplore_or_HighEffort": "PerfExp",
    "RandomSearch": "Random",
    "TuRBO": "TuRBO",
    "DRiLLS-style": "DRiLLS",
    "ROBIN-FPGA": "ROBIN",
}
METHOD_ORDER = ["Default", "PerfExp", "Random", "TuRBO", "DRiLLS", "ROBIN"]
FIG8_DESIGNS = [
    "GEMM-systolic", "Attn-head", "FFT-1024", "MobileNet-V2",
    "BFS", "PageRank", "SHA-3", "NoC-arbiter",
]


def fig8_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Figure 8: closure-rate heatmap (designs x methods)."""
    canon = df[df["phase"] == "canonical_eval_16x6x5x4"].copy()
    canon["m"] = canon["method"].map(METHODS)
    canon["closed"] = (canon["WNS_ns"] >= 0).astype(int)
    grid = (
        canon.groupby(["design", "m"])["closed"].mean().unstack()
        .reindex(index=FIG8_DESIGNS, columns=METHOD_ORDER) * 100
    )
    plt.figure(figsize=(10, 4.5))
    sns.heatmap(
        grid, annot=True, fmt=".0f", cmap="RdYlGn", vmin=20, vmax=100,
        cbar_kws={"label": "closure (%)"},
    )
    plt.title("Closure rate (%) per design × method")
    plt.tight_layout()
    plt.savefig(out / "fig08_heatmap.pdf"); plt.close()


def fig10_sigma_bars(df: pd.DataFrame, out: Path) -> None:
    """Figure 10: inter-seed sigma(WNS) bar chart."""
    canon = df[df["phase"] == "canonical_eval_16x6x5x4"].copy()
    canon["m"] = canon["method"].map(METHODS)
    designs = ["GEMM-systolic", "Attn-head", "FFT-1024", "BFS", "SHA-3", "NoC-arbiter"]
    sigma = canon.groupby(["design", "m"])["WNS_ns"].std().unstack()
    sigma = sigma.reindex(index=designs)[["ROBIN", "DRiLLS", "TuRBO", "Default"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    sigma.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(r"$\sigma$(WNS) [ns]")
    ax.set_xlabel("")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(r"Inter-seed $\sigma$(WNS) — measured")
    plt.tight_layout()
    plt.savefig(out / "fig10_sigma.pdf"); plt.close()


def fig12_coverage_curve(df: pd.DataFrame, out: Path) -> None:
    """Figure 12: conformal coverage vs target across alpha."""
    cal = df[df["phase"] == "conformal_calibration_n200"]
    ho = df[df["phase"] == "conformal_heldout_n400"]
    pred = cal.groupby("design")["WNS_ns"].mean().to_dict()

    def coverage(cal_df, ho_df, alpha):
        cr = np.abs(cal_df["WNS_ns"].to_numpy() - cal_df["design"].map(pred).to_numpy())
        n = cr.size
        q = np.sort(cr)[min(int(np.ceil((n + 1) * (1 - alpha))) - 1, n - 1)]
        hr = np.abs(ho_df["WNS_ns"].to_numpy() - ho_df["design"].map(pred).fillna(0).to_numpy())
        return float((hr <= q).mean())

    alphas = [0.20, 0.15, 0.10, 0.05, 0.03]
    targets = [1 - a for a in alphas]
    measured = [coverage(cal, ho, a) for a in alphas]

    plt.figure(figsize=(5.5, 4))
    plt.plot([0.78, 0.99], [0.78, 0.99], "k--", lw=0.6, label="ideal")
    plt.plot(targets, measured, "o-", color="#2E5C8A", label="exchangeable cal.")
    plt.xlabel(r"target coverage $1-\alpha$")
    plt.ylabel("empirical coverage")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig12_coverage.pdf"); plt.close()


def fig16_pvt_corners(df: pd.DataFrame, out: Path) -> None:
    """Figure 16: PVT corner sweep on GEMM."""
    pvt = df[df["phase"] == "extended_pvt_gemm_8corners"].copy()
    pvt["m"] = pvt["method"].map(METHODS)
    pvt = pvt[pvt["m"].isin(["ROBIN", "DRiLLS"])]
    means = pvt.groupby(["corner", "m"])["WNS_ns"].mean().unstack()
    corner_order = ["SS-40", "SS0", "SS85", "SS125", "TT25", "TT85", "FF0", "FF125"]
    means = means.reindex([c.replace("-", "m") if "-" in c else c for c in corner_order])

    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot(kind="bar", ax=ax, color=["#2E5C8A", "#C2410C"], width=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("mean WNS [ns]")
    ax.set_xlabel("PVT corner")
    ax.set_title("PVT corner sensitivity (GEMM-systolic)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out / "fig16_pvt.pdf"); plt.close()


@click.command()
@click.option("--in", "in_path", default=None, type=click.Path(),
              help="Pickled DataFrame (output of load_runs.py)")
@click.option("--archive", default="data/robin-runs-4900.zip",
              type=click.Path(), help="Or load directly from archive")
@click.option("--out", default="paper/figs/", type=click.Path(),
              help="Output directory")
def main(in_path: str | None, archive: str, out: str) -> None:
    """Regenerate every paper figure from the manifests."""
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    if in_path and Path(in_path).exists():
        console.print(f"[cyan]Loading {in_path}[/cyan]")
        df = pd.read_pickle(in_path)
    else:
        from robin.data_loader import load_archive
        df = load_archive(archive)

    console.print(f"[green]Loaded {len(df)} runs[/green]")
    for fn in (fig8_heatmap, fig10_sigma_bars, fig12_coverage_curve, fig16_pvt_corners):
        console.print(f"  Generating {fn.__name__}...")
        fn(df, out_dir)
    console.print(f"[bold]Done. Figures in {out_dir}[/bold]")


if __name__ == "__main__":
    main()

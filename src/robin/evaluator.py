"""Evaluate a trained ROBIN policy: closure rate + conformal coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from rich.console import Console

from robin.agent import DRPPOAgent, DRPPOConfig
from robin.conformal import ConformalSignoff
from robin.data_loader import load_archive

console = Console()


@dataclass
class EvaluationResult:
    design: str
    method: str
    n_runs: int
    closure_rate: float
    mean_wns: float
    sigma_wns: float
    coverage_at_95: float | None = None
    coverage_at_90: float | None = None


def evaluate_method(df: pd.DataFrame, method: str, design: str | None = None) -> EvaluationResult:
    """Compute closure rate and inter-seed sigma for a method."""
    sub = df[df["method"] == method]
    if design is not None:
        sub = sub[sub["design"] == design]
    if len(sub) == 0:
        raise ValueError(f"no runs for method={method}, design={design}")
    closure = (sub["WNS_ns"] >= 0).mean()
    return EvaluationResult(
        design=design or "ALL",
        method=method,
        n_runs=len(sub),
        closure_rate=float(closure),
        mean_wns=float(sub["WNS_ns"].mean()),
        sigma_wns=float(sub["WNS_ns"].std(ddof=1)),
    )


def evaluate_conformal(
    df_calib: pd.DataFrame,
    df_heldout: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Calibrate split-conformal on `df_calib` and measure empirical coverage on `df_heldout`.

    The "prediction" used here is the per-design mean WNS from the calibration set
    (a simple regression head substitute for the policy's slack head when the
    trained checkpoint is not loaded).
    """
    per_design_pred = df_calib.groupby("design")["WNS_ns"].mean().to_dict()
    cal_pred = df_calib["design"].map(per_design_pred).to_numpy()
    cal_obs = df_calib["WNS_ns"].to_numpy()

    signoff = ConformalSignoff(alpha=alpha)
    signoff.calibrate(cal_pred, cal_obs)

    ho_pred = df_heldout["design"].map(per_design_pred).fillna(0.0).to_numpy()
    ho_obs = df_heldout["WNS_ns"].to_numpy()
    coverage = signoff.empirical_coverage(ho_pred, ho_obs)
    return float(coverage), float(signoff.q_alpha or 0.0)


@click.command()
@click.option("--archive", default="data/robin-runs-4900.zip",
              type=click.Path(exists=True), help="Path to the runs archive")
@click.option("--alpha", default=0.05, type=float, help="Conformal miscoverage rate")
@click.option("--design", default=None, help="Optional: restrict to one design")
def main(archive: str, alpha: float, design: str | None) -> None:
    """Reproduce headline closure + coverage numbers from the archive."""
    console.print(f"[bold cyan]Loading {archive}[/bold cyan]")
    df = load_archive(archive)

    canon = df[df["phase"] == "canonical_eval_16x6x5x4"]
    methods = ["Default", "PerfExplore_or_HighEffort", "RandomSearch",
               "TuRBO", "DRiLLS-style", "ROBIN-FPGA"]
    console.print("\n[bold]Closure rate by method (canonical eval):[/bold]")
    for m in methods:
        try:
            r = evaluate_method(canon, m, design=design)
            console.print(
                f"  {m:30s}  n={r.n_runs:4d}  closure={100*r.closure_rate:5.1f}%  "
                f"mean WNS={r.mean_wns:+.3f}  sigma={r.sigma_wns:.3f}"
            )
        except ValueError as e:
            console.print(f"  {m:30s}  [yellow]{e}[/yellow]")

    cal = df[df["phase"] == "conformal_calibration_n200"]
    ho = df[df["phase"] == "conformal_heldout_n400"]
    console.print(f"\n[bold]Conformal coverage at alpha={alpha}:[/bold]")
    for a in (0.05, 0.10, 0.15, 0.20):
        cov, q = evaluate_conformal(cal, ho, alpha=a)
        console.print(f"  alpha={a:.2f}  coverage={cov:.3f}  q={q:.3f}  "
                      f"(target {1-a:.2f}, deviation {100*abs(cov-(1-a)):.1f}pp)")


if __name__ == "__main__":
    main()

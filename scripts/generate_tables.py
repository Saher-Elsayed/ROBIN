#!/usr/bin/env python3
"""Regenerate paper tables from the manifests."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

console = Console()

METHODS = {"DRiLLS-style": "DRiLLS", "TuRBO": "TuRBO", "ROBIN-FPGA": "ROBIN"}


def table_iii_workload_class(df: pd.DataFrame, out: Path) -> None:
    """Table III: closure rate by workload class."""
    canon = df[df["phase"] == "canonical_eval_16x6x5x4"].copy()
    canon["m"] = canon["method"].map(METHODS)
    canon = canon.dropna(subset=["m"])
    canon["closed"] = (canon["WNS_ns"] >= 0).astype(int)
    grid = (canon.groupby(["workload_class", "m"])["closed"].mean().unstack() * 100).round(0)
    grid = grid.reindex(columns=["ROBIN", "DRiLLS", "TuRBO"])
    out_path = out / "table_iii_workload_class.csv"
    grid.to_csv(out_path)
    console.print(grid)
    console.print(f"[green]Saved to {out_path}[/green]")


def table_v_pvt(df: pd.DataFrame, out: Path) -> None:
    """Table V: PVT corner sweep (mean WNS by corner)."""
    pvt = df[df["phase"] == "extended_pvt_gemm_8corners"].copy()
    pvt["m"] = pvt["method"].map(METHODS).fillna("Other")
    means = pvt.groupby(["corner", "m"])["WNS_ns"].agg(["mean", "std"]).round(3)
    out_path = out / "table_v_pvt.csv"
    means.to_csv(out_path)
    console.print(means)
    console.print(f"[green]Saved to {out_path}[/green]")


@click.command()
@click.option("--in", "in_path", default=None, type=click.Path())
@click.option("--archive", default="data/robin-runs-4900.zip", type=click.Path())
@click.option("--out", default="paper/tables/", type=click.Path())
def main(in_path: str | None, archive: str, out: str) -> None:
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    if in_path and Path(in_path).exists():
        df = pd.read_pickle(in_path)
    else:
        from robin.data_loader import load_archive
        df = load_archive(archive)
    console.print(f"[green]Loaded {len(df)} runs[/green]")
    table_iii_workload_class(df, out_dir)
    table_v_pvt(df, out_dir)


if __name__ == "__main__":
    main()

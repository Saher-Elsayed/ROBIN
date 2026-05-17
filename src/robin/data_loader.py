"""Parse the ROBIN 4,900-run archive into a pandas DataFrame for analysis."""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress

console = Console()


def load_archive(archive_path: str | Path) -> pd.DataFrame:
    """Load every manifest from the .zip archive into a single DataFrame.

    Each row corresponds to one P&R run; columns are the union of all
    manifest fields (flattened from the nested JSON).
    """
    archive_path = Path(archive_path)
    rows = []
    with zipfile.ZipFile(archive_path) as zf:
        members = [
            n for n in zf.namelist()
            if n.endswith("manifest.json.gz") and not n.startswith("__MACOSX")
        ]
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading manifests", total=len(members))
            for name in members:
                with zf.open(name) as fh:
                    with gzip.open(io.BytesIO(fh.read())) as gz:
                        manifest = json.loads(gz.read().decode())
                rows.append(_flatten_manifest(manifest))
                progress.update(task, advance=1)
    df = pd.DataFrame(rows)
    return df


def load_directory(root: str | Path) -> pd.DataFrame:
    """Load manifests from an unpacked directory tree."""
    root = Path(root)
    rows = []
    manifest_paths = sorted(root.rglob("manifest.json.gz"))
    with Progress() as progress:
        task = progress.add_task("[cyan]Loading manifests", total=len(manifest_paths))
        for path in manifest_paths:
            with gzip.open(path, "rb") as gz:
                manifest = json.loads(gz.read().decode())
            rows.append(_flatten_manifest(manifest))
            progress.update(task, advance=1)
    return pd.DataFrame(rows)


def _flatten_manifest(m: dict) -> dict:
    """Flatten one manifest dict to a flat row (column-per-field)."""
    row: dict = {
        "run_id": m.get("run_id"),
        "phase": m.get("phase"),
        "timestamp": m.get("timestamp"),
        "design": m.get("design"),
        "workload_class": m.get("workload_class"),
        "vendor_family": m.get("vendor_family"),
        "device": m.get("device"),
        "part": m.get("part"),
        "tool": m.get("tool"),
        "tool_version": m.get("tool_version"),
        "method": m.get("method"),
        "seed": m.get("seed"),
        "corner": m.get("corner"),
        "period_ns": m.get("period_ns"),
        "f_target_MHz": m.get("f_target_MHz"),
        "directive_bundle_hash": m.get("directive_bundle_hash"),
        "policy_checkpoint_hash": m.get("policy_checkpoint_hash"),
    }
    pr = m.get("post_route", {}) or {}
    row.update({
        "WNS_ns": pr.get("WNS_ns"),
        "TNS_ns": pr.get("TNS_ns"),
        "WHS_ns": pr.get("WHS_ns"),
        "THS_ns": pr.get("THS_ns"),
        "route_ok": pr.get("route_ok"),
        "wall_clock_s": pr.get("wall_clock_s"),
    })
    util = (pr.get("utilization") or m.get("utilization") or {})
    for k in ("LUT", "FF", "BRAM", "DSP", "URAM"):
        row[f"util_{k}_pct"] = util.get(k)
    power = (pr.get("power") or m.get("power") or {})
    row["P_static_W"] = power.get("static_W")
    row["P_dynamic_W"] = power.get("dynamic_W")
    return row


@click.command()
@click.option("--archive", required=True, type=click.Path(exists=True),
              help="Path to robin-runs-4900.zip")
@click.option("--out", default="runs_df.pkl", type=click.Path(),
              help="Output pickle path")
def main(archive: str, out: str) -> None:
    """CLI: parse the 4,900-run archive into a pandas DataFrame."""
    console.print(f"[bold]Loading {archive}...[/bold]")
    df = load_archive(archive)
    console.print(f"[green]Loaded {len(df)} runs across {df['design'].nunique()} designs[/green]")
    df.to_pickle(out)
    console.print(f"[bold]Saved DataFrame to {out}[/bold]")
    console.print("\nSummary by method:")
    closure = df.assign(closed=(df["WNS_ns"] >= 0).astype(int))
    summary = closure.groupby("method").agg(
        n=("WNS_ns", "size"),
        closure_pct=("closed", lambda s: 100 * s.mean()),
        mean_WNS=("WNS_ns", "mean"),
    ).round(2)
    console.print(summary)


if __name__ == "__main__":
    main()

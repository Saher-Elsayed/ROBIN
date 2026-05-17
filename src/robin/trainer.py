"""Top-level training loop for ROBIN.

Drives the DR-PPO agent against the FPGAEnvironment, collects K-seed/Theta-corner
rollouts, computes CVaR-shaped advantages, and updates the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import torch
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from robin.agent import DRPPOAgent, DRPPOConfig
from robin.conformal import ConformalSignoff
from robin.environment import FPGAEnvironment, ToolConfig

console = Console()


@dataclass
class TrainConfig:
    design: str
    rtl_dir: str
    constraints_file: str
    device_name: str = "VE2302"
    tool: str = "vivado"
    tool_version: str = "Vivado 2024.2"
    target_period_ns: float = 5.0
    n_episodes: int = 1200
    n_seeds: int = 5
    n_corners: int = 4
    horizon: int = 8
    beta_cvar: float = 0.2
    alpha_conformal: float = 0.05
    save_every: int = 100
    out_dir: str = "runs/default"


def train(cfg: TrainConfig) -> None:
    """Run one training session end-to-end."""
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]Training ROBIN on {cfg.design} ({cfg.device_name}, {cfg.tool})[/bold cyan]")

    seed_pool = list(range(cfg.n_seeds))
    corner_set = ["SS125", "TT25", "TT85", "FF0"][: cfg.n_corners]

    tool_cfg = ToolConfig(
        tool=cfg.tool,
        tool_version=cfg.tool_version,
        flow_dir=Path("flows") / cfg.tool,
        work_dir=out_dir / "scratch",
    )
    env = FPGAEnvironment(
        design=cfg.design,
        rtl_dir=Path(cfg.rtl_dir),
        constraints_file=Path(cfg.constraints_file),
        target_period_ns=cfg.target_period_ns,
        tool_cfg=tool_cfg,
        seed_pool=seed_pool,
        corner_set=corner_set,
        device=cfg.device_name,
    )
    agent = DRPPOAgent(DRPPOConfig(beta_cvar=cfg.beta_cvar))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Training", total=cfg.n_episodes)
        for ep in range(cfg.n_episodes):
            # In production this would run real Vivado/Quartus invocations
            # via env.reset()/step(). For the public scaffold we leave the
            # rollout call here as the integration point.
            # rollout = run_one_episode(env, agent, cfg, seed_pool, corner_set)
            # agent.update(**rollout)
            progress.update(task, advance=1)
            if (ep + 1) % cfg.save_every == 0:
                agent.save(str(out_dir / f"policy_ep{ep+1}.pt"))

    agent.save(str(out_dir / "policy_final.pt"))
    console.print(f"[green]Done. Final policy saved to {out_dir / 'policy_final.pt'}[/green]")


@click.command()
@click.option("--design", required=True, help="Design name (e.g. GEMM-systolic)")
@click.option("--rtl-path", required=True, type=click.Path(exists=True))
@click.option("--xdc", required=True, type=click.Path(exists=True))
@click.option("--device", default="VE2302")
@click.option("--tool", type=click.Choice(["vivado", "quartus"]), default="vivado")
@click.option("--target-period-ns", default=5.0, type=float)
@click.option("--episodes", default=1200, type=int)
@click.option("--seeds", default=5, type=int)
@click.option("--corners", default=4, type=int)
@click.option("--beta", default=0.2, type=float, help="CVaR confidence level")
@click.option("--alpha", default=0.05, type=float, help="Conformal miscoverage rate")
@click.option("--out", default="runs/default", type=click.Path())
def main(design, rtl_path, xdc, device, tool, target_period_ns,
         episodes, seeds, corners, beta, alpha, out) -> None:
    """CLI: train ROBIN."""
    cfg = TrainConfig(
        design=design, rtl_dir=rtl_path, constraints_file=xdc,
        device_name=device, tool=tool, target_period_ns=target_period_ns,
        n_episodes=episodes, n_seeds=seeds, n_corners=corners,
        beta_cvar=beta, alpha_conformal=alpha, out_dir=out,
    )
    train(cfg)


if __name__ == "__main__":
    main()

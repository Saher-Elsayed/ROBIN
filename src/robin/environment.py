"""Toolchain wrappers: drive AMD Vivado and Intel Quartus Prime Pro from Python.

The environment exposes an OpenAI-Gym-style interface:
    reset() -> initial state (post-synth graph + tabular features)
    step(action) -> (state, reward, done, info)

where `action` is a directive-bundle index in [0, |A|=192) and the reward is
the CVaR-shaped slack reward defined in §IV-C of the paper.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    state: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class ToolConfig:
    tool: str                       # "vivado" or "quartus"
    tool_version: str               # e.g. "Vivado 2024.2"
    flow_dir: Path                  # ROBIN/flows/{vivado|quartus}
    work_dir: Path                  # scratch dir for one run
    binary: str | None = None       # override path to vivado/quartus_sh


class FPGAEnvironment:
    """Drive one P&R run end-to-end.

    Each call to step(action) writes the directive-bundle delta into a Tcl
    fragment, invokes the vendor flow, parses the produced reports, and
    returns the resulting WNS / TNS / utilization tuple.
    """

    def __init__(
        self,
        design: str,
        rtl_dir: Path,
        constraints_file: Path,
        target_period_ns: float,
        tool_cfg: ToolConfig,
        seed_pool: list[int],
        corner_set: list[str],
        device: str,
    ) -> None:
        self.design = design
        self.rtl_dir = Path(rtl_dir)
        self.constraints_file = Path(constraints_file)
        self.target_period_ns = target_period_ns
        self.tool_cfg = tool_cfg
        self.seed_pool = seed_pool
        self.corner_set = corner_set
        self.device = device
        self._state: dict[str, Any] | None = None

    # --------------- public API ---------------

    def reset(self) -> dict[str, Any]:
        """Synthesise and extract the initial post-synth state."""
        post_synth = self._run_synthesis()
        self._state = self._extract_state(post_synth)
        return self._state

    def step(self, action: int, seed: int, corner: str) -> StepResult:
        """Apply one directive bundle and run P&R."""
        directive = self._decode_action(action)
        run_dir = self._make_run_dir(seed, corner)
        try:
            wns, tns, util, runtime, route_ok = self._run_place_route(
                directive, seed, corner, run_dir
            )
        except subprocess.CalledProcessError as e:
            return StepResult(
                state=self._state or {},
                reward=-10.0,
                done=True,
                info={"error": str(e), "route_failed": True},
            )

        reward = self._compute_reward(wns, tns, util, route_ok)
        info = {
            "WNS_ns": wns,
            "TNS_ns": tns,
            "utilization": util,
            "wall_clock_s": runtime,
            "seed": seed,
            "corner": corner,
            "directive_hash": directive.get("hash"),
            "route_ok": route_ok,
        }
        return StepResult(state=self._state or {}, reward=reward, done=False, info=info)

    # --------------- internal helpers ---------------

    def _run_synthesis(self) -> Path:
        """Invoke vivado / quartus to synthesise the design."""
        cfg = self.tool_cfg
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        if cfg.tool == "vivado":
            binary = cfg.binary or shutil.which("vivado") or "vivado"
            cmd = [
                binary, "-mode", "batch",
                "-source", str(cfg.flow_dir / "synth.tcl"),
                "-tclargs",
                "--design", self.design,
                "--rtl", str(self.rtl_dir),
                "--xdc", str(self.constraints_file),
                "--device", self.device,
                "--period", str(self.target_period_ns),
                "--out", str(cfg.work_dir / "post_synth"),
            ]
        elif cfg.tool == "quartus":
            binary = cfg.binary or shutil.which("quartus_sh") or "quartus_sh"
            cmd = [
                binary, "-t", str(cfg.flow_dir / "synthesis.tcl"),
                "--design", self.design,
                "--rtl", str(self.rtl_dir),
                "--sdc", str(self.constraints_file),
                "--device", self.device,
                "--out", str(cfg.work_dir / "post_synth"),
            ]
        else:
            raise ValueError(f"unknown tool: {cfg.tool}")

        subprocess.run(cmd, check=True, cwd=cfg.work_dir)
        return cfg.work_dir / "post_synth"

    def _run_place_route(
        self, directive: dict[str, Any], seed: int, corner: str, run_dir: Path
    ) -> tuple[float, float, dict[str, float], float, bool]:
        cfg = self.tool_cfg
        directive_tcl = run_dir / "directive.tcl"
        directive_tcl.write_text(self._directive_to_tcl(directive))

        if cfg.tool == "vivado":
            binary = cfg.binary or shutil.which("vivado") or "vivado"
            cmd = [
                binary, "-mode", "batch",
                "-source", str(cfg.flow_dir / "place_route.tcl"),
                "-tclargs",
                "--directive", str(directive_tcl),
                "--seed", str(seed),
                "--corner", corner,
                "--out", str(run_dir),
            ]
        else:
            binary = cfg.binary or shutil.which("quartus_sh") or "quartus_sh"
            cmd = [
                binary, "-t", str(cfg.flow_dir / "fit.tcl"),
                "--directive", str(directive_tcl),
                "--seed", str(seed),
                "--corner", corner,
                "--out", str(run_dir),
            ]

        import time
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        runtime = time.time() - t0
        route_ok = result.returncode == 0

        wns, tns, util = self._parse_reports(run_dir)
        return wns, tns, util, runtime, route_ok

    def _parse_reports(self, run_dir: Path) -> tuple[float, float, dict[str, float]]:
        """Parse report_timing_summary.rpt and report_utilization.rpt."""
        timing_rpt = run_dir / "report_timing_summary.rpt"
        util_rpt = run_dir / "report_utilization.rpt"
        wns = self._parse_wns(timing_rpt)
        tns = self._parse_tns(timing_rpt)
        util = self._parse_utilization(util_rpt)
        return wns, tns, util

    @staticmethod
    def _parse_wns(rpt: Path) -> float:
        if not rpt.exists():
            return -1.0
        for line in rpt.read_text().splitlines():
            if "WNS" in line and "ns" in line:
                parts = line.split()
                for p in parts:
                    try:
                        return float(p)
                    except ValueError:
                        continue
        return 0.0

    @staticmethod
    def _parse_tns(rpt: Path) -> float:
        if not rpt.exists():
            return 0.0
        for line in rpt.read_text().splitlines():
            if line.strip().startswith("TNS") or "Total Negative Slack" in line:
                for p in line.split():
                    try:
                        return float(p)
                    except ValueError:
                        continue
        return 0.0

    @staticmethod
    def _parse_utilization(rpt: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        if not rpt.exists():
            return out
        for line in rpt.read_text().splitlines():
            for key in ("LUT", "FF", "BRAM", "DSP", "URAM"):
                if line.lstrip().startswith(key) and "%" in line:
                    try:
                        pct = float(line.split("%")[0].split()[-1])
                        out[key] = pct
                    except (IndexError, ValueError):
                        pass
        return out

    def _extract_state(self, post_synth: Path) -> dict[str, Any]:
        graph_json = post_synth / "timing_graph.json"
        feature_json = post_synth / "tabular_features.json"
        if not graph_json.exists() or not feature_json.exists():
            return {"graph": None, "tabular": None}
        return {
            "graph": json.loads(graph_json.read_text()),
            "tabular": json.loads(feature_json.read_text()),
        }

    def _decode_action(self, action: int) -> dict[str, Any]:
        """Map a discrete action index in [0, 192) to a directive bundle."""
        if not (0 <= action < 192):
            raise ValueError(f"action {action} out of range [0, 192)")
        strategies = ["Default", "Aggressive", "AlternateRoutability", "ExtraTimingOpt"]
        phys_opt   = ["off", "default", "aggressive"]
        pblocks    = ["none", "central", "top_corner", "edge_band"]
        retiming   = [False, True]
        route_eff  = ["normal", "high"]
        idx = action
        s = strategies[idx % 4]; idx //= 4
        p = phys_opt[idx % 3];   idx //= 3
        b = pblocks[idx % 4];    idx //= 4
        r = retiming[idx % 2];   idx //= 2
        e = route_eff[idx % 2]
        import hashlib
        bundle = {"strategy": s, "phys_opt": p, "pblock": b, "retime": r, "route_effort": e}
        bundle["hash"] = hashlib.md5(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
        return bundle

    @staticmethod
    def _directive_to_tcl(d: dict[str, Any]) -> str:
        return "\n".join([
            f"set_property STRATEGY {d['strategy']} [current_design]",
            f"set ROBIN_phys_opt {d['phys_opt']}",
            f"set ROBIN_pblock {d['pblock']}",
            f"set ROBIN_retime {1 if d['retime'] else 0}",
            f"set ROBIN_route_effort {d['route_effort']}",
        ])

    def _compute_reward(
        self, wns: float, tns: float, util: dict[str, float], route_ok: bool
    ) -> float:
        """Eq. (3) of the paper: slack reward + TNS reward + utilization penalty."""
        if not route_ok:
            return -10.0
        period = self.target_period_ns
        w_wns, w_tns, kappa = 1.0, 0.3, 5.0
        r_wns = max(-1.0, min(1.0, wns / period))
        r_tns = -min(1.0, abs(tns) / period)
        u_max = {"LUT": 90.0, "FF": 90.0, "BRAM": 85.0, "DSP": 95.0, "URAM": 85.0}
        penalty = sum(max(0.0, util.get(k, 0.0) - u_max[k]) for k in u_max)
        return w_wns * r_wns + w_tns * r_tns - kappa * penalty / 100.0

    def _make_run_dir(self, seed: int, corner: str) -> Path:
        d = self.tool_cfg.work_dir / f"run_seed{seed}_corner{corner}"
        d.mkdir(parents=True, exist_ok=True)
        return d

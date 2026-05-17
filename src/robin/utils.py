"""Utility functions: hashing, seeding, audit-manifest construction."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def hash_dict(d: dict[str, Any]) -> str:
    """SHA-256 of a JSON-serialised dict."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's contents (8-MiB chunked)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_audit_manifest(
    design: str,
    device: str,
    tool: str,
    tool_version: str,
    seed_list: list[int],
    corner_list: list[str],
    policy_checkpoint: str,
    directive_bundle: dict[str, Any],
    wns_ns: float,
    tns_ns: float,
    utilization: dict[str, float],
    conformal_q: float,
    alpha: float,
) -> dict[str, Any]:
    """Construct a per-run audit manifest for sign-off.

    The manifest is hashed and the hash binds the policy weights, tool version,
    seed pool, corner set, directive bundle, and measured outcomes.
    """
    manifest = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "design": design,
        "device": device,
        "tool": tool,
        "tool_version": tool_version,
        "seed_list": sorted(seed_list),
        "corner_list": sorted(corner_list),
        "policy_checkpoint": policy_checkpoint,
        "policy_checkpoint_sha256": (
            hash_file(policy_checkpoint) if Path(policy_checkpoint).exists() else None
        ),
        "directive_bundle": directive_bundle,
        "directive_hash": hash_dict(directive_bundle),
        "post_route": {
            "WNS_ns": float(wns_ns),
            "TNS_ns": float(tns_ns),
            "utilization": {k: float(v) for k, v in utilization.items()},
        },
        "conformal": {"alpha": float(alpha), "q_alpha": float(conformal_q)},
    }
    manifest["manifest_hash"] = hash_dict(manifest)
    return manifest

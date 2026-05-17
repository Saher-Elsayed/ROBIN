"""Smoke tests for the data loader against the released archive."""

from pathlib import Path

import pytest

from robin.data_loader import load_archive

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "robin-runs-4900.zip"


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not present")
def test_archive_loads_4900_rows():
    df = load_archive(ARCHIVE)
    assert len(df) == 4900


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not present")
def test_archive_has_six_methods():
    df = load_archive(ARCHIVE)
    assert df["method"].nunique() >= 6
    assert {"ROBIN-FPGA", "DRiLLS-style", "TuRBO", "Default"}.issubset(set(df["method"].unique()))


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not present")
def test_archive_phases_cover_paper():
    df = load_archive(ARCHIVE)
    phases = set(df["phase"].unique())
    expected = {
        "canonical_eval_16x6x5x4",
        "training_sweep_dr_ppo",
        "conformal_heldout_n400",
        "tool_version_stress",
        "extended_pvt_gemm_8corners",
        "conformal_calibration_n200",
    }
    assert expected.issubset(phases)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not present")
def test_robin_closure_rate_matches_paper():
    df = load_archive(ARCHIVE)
    canon = df[df["phase"] == "canonical_eval_16x6x5x4"]
    robin = canon[canon["method"] == "ROBIN-FPGA"]
    drills = canon[canon["method"] == "DRiLLS-style"]
    robin_closure = (robin["WNS_ns"] >= 0).mean()
    drills_closure = (drills["WNS_ns"] >= 0).mean()
    # Paper claims ROBIN ~90.3%, DRiLLS ~75.9%, gap ~14.4 pp.
    assert robin_closure > drills_closure
    assert (robin_closure - drills_closure) > 0.10

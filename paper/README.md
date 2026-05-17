# ROBIN Paper

`robin-fpga.tex` is the IEEEtran source for the paper:

> S. Elsayed, **"ROBIN: Robust FPGA Timing Closure with Distributionally-Robust RL and Conformal Sign-off,"** *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2026.

## Building

```bash
make paper        # from repo root
# or
cd paper && pdflatex robin-fpga.tex && pdflatex robin-fpga.tex
```

Requires `texlive-latex-base`, `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-publishers` (for IEEEtran), `texlive-pictures`, and `texlive-science` (pgfplots ≥ 1.18).

## Regenerating figures from the dataset

Every figure in `robin-fpga.tex` is reproducible from the 4,900-run manifests in `../data/robin-runs-4900.zip`:

```bash
cd ..
python scripts/load_runs.py --archive data/robin-runs-4900.zip --out /tmp/runs.pkl
python scripts/generate_figures.py --in /tmp/runs.pkl --out paper/figs/
python scripts/generate_tables.py  --in /tmp/runs.pkl --out paper/tables/
```

Outputs:

| Figure / Table                          | Script                       |
| --------------------------------------- | ---------------------------- |
| Fig. 8 (heatmap: closure × method)      | `generate_figures.py`        |
| Fig. 10 (σ-bar chart)                   | `generate_figures.py`        |
| Fig. 12 (conformal coverage)            | `generate_figures.py`        |
| Fig. 16 (PVT corner sweep)              | `generate_figures.py`        |
| Table III (workload-class closure)      | `generate_tables.py`         |
| Table V (PVT corner means)              | `generate_tables.py`         |

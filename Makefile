.PHONY: help install dev test lint format paper figures tables clean

help:
	@echo "ROBIN — common targets"
	@echo "  make install   install package"
	@echo "  make dev       install package with dev extras"
	@echo "  make test      run pytest"
	@echo "  make lint      black + ruff + mypy"
	@echo "  make format    auto-format with black + ruff --fix"
	@echo "  make paper     rebuild paper/robin-fpga.pdf"
	@echo "  make figures   regenerate paper figures from data/robin-runs-4900.zip"
	@echo "  make tables    regenerate paper tables"
	@echo "  make clean     remove build artefacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks]"

test:
	pytest -v --cov=robin --cov-report=term-missing

lint:
	black --check src tests scripts
	ruff check src tests scripts
	mypy src

format:
	black src tests scripts
	ruff check --fix src tests scripts

paper:
	cd paper && pdflatex -interaction=nonstopmode robin-fpga.tex && pdflatex -interaction=nonstopmode robin-fpga.tex

figures:
	python scripts/load_runs.py --archive data/robin-runs-4900.zip --out /tmp/runs_df.pkl
	python scripts/generate_figures.py --in /tmp/runs_df.pkl --out paper/figs/

tables:
	python scripts/generate_tables.py --in /tmp/runs_df.pkl --out paper/tables/

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	cd paper && rm -f *.aux *.log *.out *.toc *.bbl *.blg
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

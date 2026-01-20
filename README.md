# ClinicAlly

Codebase for the paper "ClinicAlly: Integrating Causal Representation Learning with Neuro-Symbolic Reasoning for Interpretable Clinical Decision Support".

## Structure
- `model.py`: Core implementation of the ClinicAlly architecture.
- `run_experiments_v3.py`: Experiment runner for benchmarking.
- `optimize_clinically.py`: Script for optimizing ensemble weights.
- `fetch_dataset.py`: Utilities for handling the MIMIC-IV dataset.
- `v3.tex`: LaTeX source of the paper.
- `update_v3_tex.py`: Helper script to update results in the paper automatically.

## Usage
1. Download MIMIC-IV demo data.
2. Run `python fetch_dataset.py`.
3. Run `python run_experiments_v3.py`.

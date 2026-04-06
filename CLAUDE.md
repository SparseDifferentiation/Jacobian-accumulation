# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Benchmarking `SparseDiffEngine` against ADOL-C, JAX, and CasADi for computing Jacobians and other derivatives. The focus is on understanding the performance tradeoffs of approaches like graph coloring, Jacobian accumulation, and vertex elimination for exploiting sparsity patterns.

## Environment Setup

- Python 3.13 virtualenv at `.venv/`
- Activate: `source .venv/bin/activate`
- Installed packages: `casadi`, `numpy`

## Running Scripts

```bash
source .venv/bin/activate
python casadi/casadi_hello_world.py
python jax/jax_hello_world.py
```

## Repository Structure

Each differentiation tool has its own directory (`adol-c/`, `casadi/`, `jax/`) containing benchmark scripts. The project is in early development — `adol-c/` and `jax/` are mostly stubs.

# 🧮OpenIsing

This repository aims at exploring different flavors of Ising model solvers with the overarching goal of developing
on-chip Ising machines. The codebase serves as a platform for testing, benchmarking, and evaluating various algorithms
and strategies in software and hardware.

## This Fork

This fork of OpenIsing aims to improve dependency management, speed and introduce a place for shared, reproducible experiments that can be shared.

The repository is structured as follows:

```
├── data                # for experiment output (.csv, .pkl etc..)
├── ising               # core ising package
├── notebooks           # for marimo notebooks
└── Paper_OpenIsing    # original paper
```

## Usage

### Environment

For dependency management [uv](https://docs.astral.sh/uv/) is used. It is super fast and highly recommended. All dependencies are specified in `pyproject.toml`. For installation of `uv` see the [official docs](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is installed, generate the virtual environment with:

```bash
uv sync
```

for adding new packages to the environment:

```bash
uv add my_favorite_package
```

### Notebooks

For experiments and one-off code it is encouraged to use notebooks. The repository is set up to use [marimo](https://marimo.io/). It is very similar to Jupyter but has some very nice additional features. It is already part of the environment. The recommended way to work is to install the [official vscode extension](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo) to use the notebooks directly in VSCode. If preferred, marimo can also host a web server like Jupyter.

Marimo has many advantages, but it's main (my favorite) advantage is that the file is pure Python with some added decorators. This is nice for version control and also allows us to run a notebook as if it was just a normal Python script.

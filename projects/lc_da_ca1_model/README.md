# LC-DA-CA1 Standalone Model

This folder is a standalone extraction of the synthetic LC-DA-CA1 model from the main research archive. It is intended to be runnable on any machine without access to the lab network drive or the larger archive utilities.

## What This Is

- `run_model.py`: standalone model runner that generates the full figure set.
- `support/`: minimal vendored utility modules required by the model.
- `outputs/`: default output directory for generated figures.

## Quick Start

Create an environment and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight smoke pass:

```bash
python run_model.py --n-bootstrap 2 --n-cells 200
```

Run the default publication-style configuration:

```bash
python run_model.py
```

Generated figures are saved under `outputs/` by default. You can override that location:

```bash
python run_model.py --output-dir ./my_outputs
```

Or via environment variable:

```bash
LC_DA_CA1_OUTPUT_DIR=./my_outputs python run_model.py
```

## Why This Extraction Exists

The main repository is a research archive. This folder provides a curated, portable unit with a smaller dependency surface and no reliance on lab-only data paths.

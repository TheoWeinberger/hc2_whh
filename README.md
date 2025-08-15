# WHH Multi-Band Hc2 Fitting Tool

## Overview
This program fits upper critical field \( H_{c2}(T) \) data to Werthamer–Helfand–Hohenberg (WHH) models for one-band and two-band superconductors. It supports:
- **Orbital-limited** models
- **Pauli-limited** models (with Maki parameter \( \alpha \) and spin–orbit scattering \( \lambda_{\text{so}} \))
- **Two-band** orbital and Pauli-limited models
- Symmetric or asymmetric interband coupling

The fitting enforces **monotonicity** of \( H_{c2}(T) \) and can automatically **fall back** to stable orbital fits before attempting Pauli-limited fits.

---

## Installation
Requires:
- Python ≥ 3.9
- `numpy`, `pandas`, `matplotlib`, `scipy`

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy
```

---

## Usage, Modes, Examples, and Options

### Command Syntax
```bash
python hc2_multifit.py "<file1.csv>, <file2.csv>, ..." <mode> [options]
```
- `<file1.csv>, <file2.csv>, ...` — One or more CSV/TXT files containing `T` and `H` columns.
- `<mode>` — Model type to fit (see below).

### Modes
- `orbital` — Single-band orbital-limited WHH model.
- `pauli` — Single-band Pauli-limited WHH model with Maki parameter and spin–orbit scattering.
- `2band_orbital` — Two-band orbital-limited WHH model.
- `2band_pauli` — Two-band Pauli-limited WHH model.

### Examples
```bash
# Fit a single-band Pauli-limited model to one dataset
python hc2_multifit.py "sc33.txt" pauli

# Fit two datasets to a two-band Pauli-limited model with symmetric couplings
python hc2_multifit.py "sc11.txt, sc33.txt" 2band_pauli

# Fit two-band Pauli-limited with independent λ12 and λ21
python hc2_multifit.py "sc11.txt, sc33.txt" 2band_pauli --untie_interband

# Fit single-band Pauli-limited with fixed Tc and α
python hc2_multifit.py "sc33.txt" pauli --Tc_fixed=1.65 --alpha=0.8
```

### Options
#### General
- `--Tc_fixed=<K>` — Fix \( T_c \) instead of fitting it.
- `--untie_interband` — In two-band fits, allow λ₁₂ and λ₂₁ to be fitted independently.

#### Single-Band Parameters
- `--alpha=<value>` — Fix Maki parameter \( \alpha \).
- `--lambda_so=<value>` — Fix spin–orbit scattering \( \lambda_{\text{so}} \).
- `--alpha_bounds=lo,hi` — Bounds for \( \alpha \).
- `--lambda_bounds=lo,hi` — Bounds for \( \lambda_{\text{so}} \).

#### Two-Band Coupling Parameters
- `--lam11=<value>` — Fix intraband coupling λ₁₁.
- `--lam22=<value>` — Fix intraband coupling λ₂₂.
- `--lam12=<value>` — Fix interband coupling λ₁₂.
- `--lam21=<value>` — Fix interband coupling λ₂₁ (ignored in symmetric mode).

#### Two-Band Shared Parameters
- `--eta=<value>` — Fix diffusivity ratio \( D_2 / D_1 \).

#### Two-Band Pauli Parameters
- `--alpha1=<value>` — Fix Maki parameter for band 1.
- `--alpha2=<value>` — Fix Maki parameter for band 2.
- `--lambda_so1=<value>` — Fix spin–orbit parameter for band 1.
- `--lambda_so2=<value>` — Fix spin–orbit parameter for band 2.
- Bounds can be set via:
  - `--alpha1_bounds=lo,hi`
  - `--alpha2_bounds=lo,hi`
  - `--lambda_so1_bounds=lo,hi`
  - `--lambda_so2_bounds=lo,hi`

---

## Output
For each input dataset:
- `<tag>_<mode>_results.csv` — Columns: `T_K`, `Hc2_T_data`, `Hc2_T_fit_mono`.
- `<tag>_<mode>_params.csv` — Fitted parameters and metrics (R², RMSE, MAE, etc.).

For all datasets combined:
- `combined_<mode>_params.csv` — Fitted parameters for all datasets.
- `combined_<mode>_plot.pdf` — Publication-quality plot from \( T = 0 \) to \( T_c \).

---

## Stability
If a two-band Pauli fit produces unphysical parameters:
1. The program first tries an orbital fit to seed parameters.
2. If instability persists, it will print recommended λ-values to fix in a rerun:
   ```bash
   python hc2_multifit.py "sc11.txt, sc33.txt" 2band_pauli --lam11=... --lam22=... --lam12=...
   ```

---

## Citation
If you use this in research, please cite:
- N.R. Werthamer, E. Helfand, and P.C. Hohenberg, *Phys. Rev.* **147**, 295 (1966).
- A. Gurevich, *Phys. Rev. B* **67**, 184515 (2003).
- A. Gurevich, *Phys. Rev. B* **82**, 184504 (2010).
- T. I. Weinberger et al. *arXiv* **2505.12131** (2025).

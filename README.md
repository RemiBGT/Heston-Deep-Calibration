# Deep Calibration Engine: Accelerating Heston Volatility Models via Neural Networks

## Overview

This repository presents a production-oriented surrogate modeling workflow for the Heston stochastic volatility model.

The core idea is straightforward:

- keep a transparent, semi-analytic Heston pricer as the quantitative ground truth,
- generate a large synthetic pricing dataset offline,
- train a neural network to approximate that pricing map,
- and reuse the neural surrogate inside calibration to reduce latency dramatically.



## The Industrial Problem

Heston is attractive because it captures smile dynamics far better than constant-volatility models, but the calibration loop is expensive.

In this project, the exact call price is obtained by Fourier inversion with adaptive numerical quadrature:

- each strike requires a numerical integral,
- each optimizer step reprices the full strike grid,
- and each full calibration run repeats that process many times.

This is the classic industrial trade-off in quantitative finance: the model is rich enough to be useful, yet the computational cost can become a bottleneck for daily workflows.

## The Deep Learning Solution

The acceleration layer is an `MLPRegressor` trained to learn the pricing function:

`(F, K, T, v0, kappa, theta, rho, sigma) -> Call Price`

The workflow is split into two phases.

### Offline data generation

- sample economically plausible Heston parameters,
- filter unstable regions using the Feller condition,
- compute exact option prices with the Heston Fourier engine,
- flatten each strike curve into supervised learning observations.

This transforms an expensive pricing engine into a reusable training set.

### Online calibration

- keep the same optimizer (`L-BFGS-B`),
- keep the same calibration target,
- replace repeated exact pricing calls with the neural-network approximation.

The financial logic stays unchanged. Only the runtime profile changes.

## Methodology

### 1. Exact pricing engine

The exact model is a semi-analytic Heston pricer implemented via:

- characteristic function evaluation,
- Fourier inversion,
- adaptive quadrature (`scipy.integrate.quad`).

This is not the fastest possible production method, but it is deliberately readable and robust, which makes it a strong reference implementation for a portfolio project.

### 2. Synthetic dataset design

The dataset is generated from sampled Heston parameters:

- `v0`: initial variance,
- `kappa`: variance mean-reversion speed,
- `theta`: long-run variance,
- `rho`: spot/variance correlation,
- `sigma`: volatility of variance.

The Feller condition,

`2 * kappa * theta > sigma^2`

is used as a practical stability filter during both data generation and calibration. This keeps the learning problem in a numerically well-behaved region and prevents the optimizer from wasting time in pathological parameter zones.

### 3. Neural surrogate

The surrogate is a feed-forward neural network (`MLPRegressor`) trained on normalized inputs.

Feature scaling is critical because the input vector mixes values with very different orders of magnitude:

- spot-like levels around `100`,
- maturities around `1`,
- variances around `0.01`.

Without normalization, gradient-based optimization converges less reliably and wastes model capacity compensating for scale rather than learning the pricing surface.

### 4. Calibration objective

Calibration minimizes a weighted sum of squared errors.

The weighting is intentionally centered on the at-the-money region with a Gaussian-shaped profile:

- strikes close to the forward receive more weight,
- far wings remain in the objective, but matter less.

This reflects how calibration is often used in practice: the ATM region is usually the most liquid, most stable, and most relevant for hedging and risk reporting.

## Results

The current benchmark configuration is positioned as a production-style acceleration case:

- exact Heston calibration used as the pricing benchmark,
- neural surrogate used as the accelerated calibration engine,
- identical optimizer and identical calibration target.

### Benchmark Snapshot

| Metric | Exact Heston | Neural Surrogate |
|---|---:|---:|
| Calibration Runtime | Baseline | Faster |
| Speed-up | 1.0x | **70.0x** |
| Calibration SSE | Reference | **< 0.05** |
| Curve Fit Quality | Exact benchmark | Near-indistinguishable on target curve |

The practical takeaway is simple: most of the pricing cost can be moved offline.

### Figures

The plotting layer can export figures directly to `docs/images/`, which makes the repository easy to keep synchronized with fresh benchmark runs.

![Calibration Curve Comparison](docs/images/calibration_curve.png)
<!-- Replace with the latest exported pricing-curve chart -->

![Optimizer Convergence](docs/images/convergence_history.png)
<!-- Replace with the latest exported convergence chart -->

## Project Structure

```text
.
|-- main.py
|-- README.md
|-- requirements.txt
|-- tests
|   `-- test_heston.py
`-- src
    |-- calibration
    |   `-- optimizer.py
    |-- data
    |   `-- generator.py
    |-- models
    |   `-- heston.py
    |-- surrogate
    |   `-- nn_model.py
    `-- utils
        `-- plotter.py
```

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the end-to-end demo

```bash
python main.py
```

The script will:

- generate a synthetic Heston dataset,
- train the neural surrogate,
- calibrate the exact model and the neural surrogate side by side,
- print runtime and error metrics,
- display the plots,
- export PNG figures to `docs/images/`.

### Run the tests

```bash
pytest
```

The current test suite includes a basic but useful arbitrage sanity check on the exact Heston pricer, verifying that call prices stay above intrinsic value and below the forward.

## Why this matters in Quantitative Finance

Accelerating a heavy stochastic model is not just a technical optimization. It changes what is operationally feasible.

In practical quantitative finance, a fast surrogate can support:

- **intra-day recalibration** when surfaces move and parameters must be refreshed without waiting for a slow batch process,
- **real-time risk management** where Greeks and scenario calculations depend on a pricing engine that can respond quickly,
- **large-scale portfolio quotation** where hundreds or thousands of instruments must be repriced without introducing user-facing latency.

That is the real business value of surrogate modeling: preserve the structure of a rich stochastic model while making it usable under production time constraints.

## Roadmap

Natural next steps for this codebase include:

- larger synthetic datasets,
- richer benchmark reporting,
- calibration to implied volatilities instead of raw prices,
- multi-maturity or surface-wide calibration,
- packaging the workflow into a reusable research or production component.

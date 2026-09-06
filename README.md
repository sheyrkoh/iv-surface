# iv-surface

Constructing an implied volatility surface from SPX index options data.

This repository builds a volatility surface end to end: a Black–Scholes pricer, a
numerical implied volatility solver, a documented quote-cleaning layer, forward and
discount factor extraction from market prices, and arbitrage diagnostics on the
resulting surface.

It is a work in progress. The status table below is accurate — nothing is described
here as working unless it is implemented and tested.

## Status

| Phase | Description | State |
|-------|-------------|-------|
| 0 | Repository and environment | Complete |
| 1 | Black–Scholes pricer and Greeks | Complete |
| 2 | Implied volatility solver | Complete |
| 3 | Data ingestion (Cboe delayed quotes → Parquet) | Not started |
| 4 | Quote cleaning and filter attrition | Not started |
| 5 | Forward and discount factor extraction | Not started |
| 6 | Surface construction | Not started |
| 7 | Arbitrage checks | Not started |
| 8 | Analytical question | Not started |
| 9 | Write-up | Not started |

## What is implemented

### `src/pricer.py`

European call and put pricing under Black–Scholes, plus delta, gamma, vega, theta
and rho.

Unit conventions are fixed and documented in the module docstring, because Greeks
are quoted inconsistently across sources and an unstated convention is a bug waiting
to happen:

| Greek | Convention |
|-------|------------|
| vega | per 1 percentage point of volatility (analytic derivative / 100) |
| theta | per calendar day (analytic derivative / 365) |
| rho | per 1 percentage point of the rate (analytic derivative / 100) |
| theta sign | `dV/dT` with `T` = time to expiry, **not** negated decay-per-day |

`gamma` and `vega` take no `option_type` argument. They are identical for calls and
puts, which follows immediately from differentiating put–call parity — the
difference between a call and a put is linear in spot and independent of volatility.

### `src/iv_solver.py`

Inverts the pricer to recover implied volatility from a market price.

- **No-arbitrage bounds** (`price_bounds`) derived from put–call parity, giving the
  achievable price interval for a given contract. Prices at or outside those bounds
  are rejected before any solver runs.
- **Newton–Raphson fast path** using vega as the analytic derivative.
- **Brent fallback** (`scipy.optimize.brentq`) when Newton leaves the bracket.
  Overshooting is a failure of the method, not of the problem, so it is worth
  retrying with a bracketed solver.
- **Vega-collapse check.** If vega falls below a threshold, the solver returns `NaN`
  immediately rather than falling through to Brent. Near-zero vega means the price
  is insensitive to volatility — that is a property of the contract, and no solver
  can rescue it.
- **Never raises.** Every failure path returns `NaN` and logs a distinct reason, so
  failures can be counted and categorised in the Phase 4 attrition table rather than
  vanishing.

## Design decisions

**SPX, not SPY.** SPX index options are European-exercise and cash-settled, so
Black–Scholes applies directly. SPY options are American-style; using them means
either modelling the early-exercise premium or silently getting the wrong implied
volatility. The choice costs nothing and removes a whole class of error.

**Forwards extracted, not assumed.** Rather than plugging in a risk-free rate and a
dividend yield, the forward and discount factor for each expiry will be recovered by
regressing (call − put) against strike across the chain — put–call parity read as a
straight line. This removes two arbitrary inputs.

**Filter attrition is recorded, not hidden.** Every cleaning rule will report how
many contracts it removes. A surface built from filtered data is only as credible as
the filtering, and the counts are the evidence.

**Modules with tests, not a notebook.** Logic lives in `src/` with `pytest` coverage.
A single demonstration notebook will come later.

## Repository layout

```
src/          pricer, solver, and (later) ingestion, cleaning, surface modules
tests/        pytest suite
data/         cached Parquet snapshots (not version-controlled)
notebooks/    demonstration notebook (later)
```

## Setup

Requires Python 3.11 or later.

```powershell
git clone https://github.com/sheyrkoh/iv-surface.git
cd iv-surface
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Tests

```powershell
pytest
```

Current suite: 11 tests.

**Pricer** — put–call parity to 1e-10; deep-in-the-money calls converge to
discounted intrinsic value; vega is strictly positive everywhere and peaks near the
forward. That last test is not decoration: implied volatility is only a well-posed
inverse problem because the price is strictly monotonic in volatility, and positive
vega is exactly that condition.

**Solver** — recovery of a known volatility through the Newton path and through the
Brent fallback path separately (the fallback case is constructed to make Newton
overshoot, so the branch is genuinely exercised); recovery on the put side;
rejection at and beyond both no-arbitrage bounds; `NaN` return on degenerate vega.

## Data

Primary source is Cboe delayed quotes for SPX. One snapshot per trading day taken
after the US close, saved to Parquet with a timestamp, building a local history.

Quotes are delayed and are used for surface construction and analysis only. No
credentials are stored in this repository.

## Licence

MIT.

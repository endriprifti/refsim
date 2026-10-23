# Referendum Design Simulator (`refsim`), version 1.1.5

The reusable tool of *Engineering Albania's Referendum Law: A Design Guide*, Section VI. It computes, for any validity rule, referendum type, calendar and verification rule, the four criteria of Section I: C1 and C1′ (false invalidation and false validation), C2 (status-quo capture), C3 (contamination), C4 (accessibility) and C4′ (false admission). It does so with the equilibrium model of Sections III–V, under data that a drafter can update.

Every number in Sections III–V that the simulator recomputes is pinned by a regression test (`tests/`, 111 tests). Where the simulator differs from the printed text, the reason is a modelling decision of the book (D10 exact ρ; D11 adult drift rate and 2025-basis municipal proxies); `docs/ASSUMPTIONS.md` lists every such case.

## Quick start

```bash
pip install -r requirements.txt          # numpy, scipy, pyyaml, matplotlib, python-dateutil, pytest
python -m pytest -q                       # 111 regression tests, about 15 s
python run.py reproduce                   # key values of Sections III-V side by side with the book
python run.py all                         # every run below, about 8 minutes on two cores
```

| Command | What it produces (in `outputs/<scenario>/`) |
|---|---|
| `run.py reproduce` | `reproduce.csv`: simulator against printed values |
| `run.py packages` | `packages_central.csv`, `packages_uncertainty.csv` (2.5–97.5% intervals, minimum and maximum), `packages_draws.csv`, `packages_paired.csv` (paired differences against P0 on common draws), `local_units.csv`, `local_summary.csv`, Figure 6.3 |
| `run.py grid` | `menu_grid.csv`: 19 menu rules × D3 × D9 × π × denominator (1,596 cells), Figure 6.6 |
| `run.py maps` | regime maps over ρ and τ, prior shape a_B, drift rate d_R, campaign length L and complexity k, T_rel and the window; Figures 6.1, 6.2, 6.4, 6.5 |
| `run.py types` | rules in force by type, T1–T6, over D3 × D9 |
| `run.py signatures` | C4′ and false rejection: the current 5% sample against the F44 band rule |
| `run.py pipeline` | Pr_vote within 12/24/48 months by timing scenario TS0–TS9, prompt or latest President, verification rule |
| `run.py convergence` | Monte Carlo (≥ 10,000 replications per cell, ex-post rule evaluation) against exact quadrature |
| `run.py sensitivity` | turnout-noise sweep, Pareto map with noise, T4 weekday shares, literal national-register reading for T7 and T8, lock boundaries of the April 2026 draft |
| `run.py cases` | Section VII tool runs R1–R10: `vii_R*.csv`; digest `docs/CASES_DIGEST.md` (`scripts/make_cases_digest.py`) |
| `run.py calibrated --scenario calibrated_2026` | Section VII Part 2 reruns TR11–TR20: `vii2_TR*.csv`; digest §§32–43 |
| `run.py design --scenario calibrated_2026` | Section VIII tool runs TR21–TR28 and Appendix H's TR29: `viii1_TR*.csv`, `viii2_TR29_*.csv`; digest §§44–52 |
| `run.py accession --scenario calibrated_2026 --means 0.6,0.7,0.8` | TR29 only: w* against V1 for T6 under accession priors (add the author's mean) |

Every CSV has a `.meta.json` sidecar: the scenario file, the data vintage, the run time, and every parameter value with its source and date.

## Layout

```
config/       baseline_2025.yaml (every parameter: value, source, date), types.yaml (T1-T8), packages.yaml (P0-P5)
data/         Table C.4 municipal proxies (61 municipalities, Law No. 115/2014; and 46, Law No. 100/2026), Assembly cycles, local elections, INSTAT population
refsim/       the package (twelve modules of the Section VI specification)
  rules/      validity rules V1-V11: one plug-in each, common interface
  electorate  denominators, effective bars, municipal proxies, rebasing (D11)
  priors      A2 uniform, A12 Beta, zealot family (5.27), kappa calibration
  game        equilibrium engine (pure; approximate mixed), criteria with A11 noise
  simulate    Monte Carlo on vote counts, Wilson intervals, convergence check
  pipeline    wraps the checked lifted chain of Section V Part 1; C4
  calendar    windows, one annual day, cut-off, repeat calendar (H1, BP/BF, VR)
  campaign    learning model (Proposition 24, Corollary 7)
  signatures  R0-R5 verification rules, exact errors
  opinion     voter model, bar vs fidelity, same-issue test, Jacobi diffusion (wraps B.51)
  window      OU premium, window tests, two-point bound, Table V.24 construction
  drift       A29 path, lock years, inverse-Gaussian bands, municipal drift
  review      indicators I1-I10, trigger (5.37), even-year schedule
  regime_b    Proposition 30 (Regime B under general priors)
  contamination, v10, evaluate, runs, report
legacy/       the checked code archives of Sections IV-V and Consistency check 2, unchanged
tests/        regression and unit tests
scripts/      update_data.py: new register, census and election figures -> new scenario
docs/         MANUAL.md (Appendix G), ASSUMPTIONS.md, PROPOSITION30.md
```

## Versions

- 1.1.5 (23 October 2026): data and one option. (1) The municipal map of Law No. 100/2026 (adopted 9 October 2026; 46 municipalities from the 2027 local elections, the 12 qarqe unchanged): `data/table_c4_municipalities_2027.csv`, aggregated from Table C.4 by whole-municipality mergers within the same qark (column `members`). It is the default map; `run.py ... --map 2014` (or `REFSIM_MAP=2014`) reproduces the 61-unit results of v1.1.4 into `outputs/<scenario>_map2014/`. `run.py local` reruns everything that uses the map: the local units of the packages, the municipal drift, the literal national-register reading and R10 (baseline), and TR18, TR25 and TR30(b) (`--scenario calibrated_2026`). Qark results are unchanged, since the qarqe keep their composition. (2) R3 (1998): observed turnout 0.5057, the Central Voting Commission's 50.57% of 27 November 1998 (968,346 of 1,914,859; 0.5053 before). (3) R10: the municipal rule's draws are padded to the 61-unit count, so that the shared random stream, and with it every P4 row, is the same on both maps (the 2014 map reproduces v1.1.4 exactly). Tests updated for both maps.
- 1.0 (Step 13): the simulator, 84 regression tests, the baseline run.
- 1.0.1 (revision VI-1): outputs only — minima, maxima and paired differences over the draws; the sensitivity mode. Every 1.0 number unchanged.
- 1.0.2 (revision VI-2): outputs only — the literal national-register reading for T8; digest sections 20 (menu by π) and 21 (local units, unit by unit, for Appendix D), both from existing CSV files; corrected labels in `config/packages.yaml` (Art. 151(1), F21, for P2 and P3) and a rounding in `docs/PROPOSITION30.md` (K_F = 0.3165). Every earlier number unchanged.
- 1.0.3 (Section VII, Part 1): outputs only — tool runs R1–R10 (`refsim/cases.py`, `run.py cases`, `docs/CASES_DIGEST.md`); the combined-limb rule V8x and packages P3b, P3c (the March 2026 draft re-read); the A5 co-scheduled game (GameCo); the Italian abrogative series in `data/cases/`; 4 new tests (88 in all). Every earlier number unchanged.
- 1.1.4 (23 October 2026): data only — the 2025 in-country ballots are 1,463,314 and all ballots 1,665,501 (OSCE/ODIHR 2025 final report, Annex I, CEC results; 1,458,251 and 1,664,873 before, from press data), so the in-country anchor of the T6 accession runs (TR23, TR29) is τ = 0.394 (`design.TAU_ANCHOR`; 0.393 before). Digest §§46 and 52 change in that column only; τ's calibrated range is unchanged at two decimals; `tauR_ref` stays 0.76. Every other number unchanged.
- 1.1.3 (Section IX, 15 October 2026): tool runs TR30–TR32 (`refsim/robustness.py`, `run.py robustness --scenario calibrated_2026`, digest §§53–55): the boundary between V1 and V7 at 0.55 across the 15 one-at-a-time variants, for T3–T5 and every local unit, and under unequal stakes (Ω = 1/2, 2); TS9 with the 91-day lead of voting from abroad and the T5 delay; V1 with votes from abroad (τ_ab, μ_A); the first-year cost of Pv61 for the 2026 initiatives. One new argument, the stake ratio `Omega` of `evaluate.cell` (default 1: every earlier number unchanged). 7 new tests (109 in all). §§32–52 unchanged.
- 1.1.2 (Appendix H, 14 October 2026): outputs only — tool run TR29 (`refsim/design.py: run_TR29`, `run.py accession`, digest §52): the indifference weight w* against V1 for T6 under the illustrative accession priors; the manual (Appendix G) updated for the calibrated scenario (base scenario, commands, run-to-file map, parameters without a flag, TS9 as computed, cross-references to Appendix H). §§32–51 unchanged; 102 tests.
- 1.1.1 (Section VIII, Part 1): tool runs TR21–TR28 (`refsim/design.py`, `run.py design --scenario calibrated_2026`, `docs/CALIBRATED_DIGEST.md` §§44–51): validity and procedure combined; the binding-effect screen of the T1 menu; the menu for T3, T4, T6 (and T6 under illustrative accession priors); the T5 menu; the local menu with proxy error; the window at the calibrated decay constants; level boundaries by turnout; the complexity bound under V1 and V7. Corrections: V10 is labelled with the σ_u actually used (values unchanged); in calibrated_2026 the turnout noise of a rule counted on residents is σ_τ·ρ (scenario switch `noise_scaled_on_R`; P1's T5 row moves from 0.341 to 0.327 at the centre; baseline_2025 keeps the v1.0 convention). `cell()` accepts an explicit prior; `evaluate_package()` accepts a package dict and a President setting. 6 new tests (99 in all). baseline_2025 unchanged (largest difference 4.7 × 10⁻⁷, solver tolerance).
- 1.1.0 (Section VII, Part 2): the calibrated scenario `config/calibrated_2026.yaml` and reruns TR11–TR19 (`refsim/calibrated.py`, `run.py calibrated --scenario calibrated_2026`, `docs/CALIBRATED_DIGEST.md`). Model options, all off in baseline_2025: A36 (a default side under a participation-type rule may call to abstain, floor `t_min_boycott`); `t_bar` passed from the scenario; π, μ_b and Ψ by type with the government's side; prior families `beta_shift` (D13: Beta with the variance of Beta(a_B, a_B) and the A2 type mean) and `zealot_vm` (zealot family with a_p(π) by (7.6)). 5 new tests (93 in all). Every baseline_2025 number unchanged (largest difference in a package criterion 3.6 × 10⁻⁷, solver tolerance).

## Reading the results

Start with `docs/MANUAL.md`, section 6 ("How to read regime maps"). The headline results of the baseline run are summarised in `docs/RESULTS_DIGEST.md`, which Section VI-b uses.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The files in `data/` reproduce published figures of the Central Election Commission (KQZ), OSCE/ODIHR and INSTAT and keep their sources' terms: INSTAT's official statistics may be reused with the citation "Burimi: INSTAT" and not for commercial gain; OSCE/ODIHR texts may be used for educational purposes with credit; the KQZ publishes no reuse terms, so its figures are reproduced with their source.

Cite as: Prifti, E. (2026), *Referendum Design Simulator (refsim)*, version 1.1.5 [software], companion to *Engineering Albania's Referendum Law: A Design Guide*. Each release is archived on Zenodo, which assigns it a DOI; the DOI is shown on the Zenodo record linked from this repository. `CITATION.cff` holds the reference, and GitHub shows it under "Cite this repository".

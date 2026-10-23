# Appendix G — Manual of the Referendum Design Simulator

This manual (refsim v1.1.5) is written for a drafter, a clerk of a parliamentary committee, or an election-administration analyst who wants to rerun the analysis of this book with new data or test a rule the book did not consider. It assumes no programming beyond running commands and editing a text file.

## 1. Installation

1. Install Python 3.11 or later.
2. Unpack the archive and open a terminal in its folder.
3. Install the libraries: `pip install -r requirements.txt`.
4. Check the installation: `python -m pytest -q`. All 109 tests should pass in about twenty seconds (84 in v1.0, 93 in v1.1.0, 99 in v1.1.1, 102 in v1.1.2). If one fails, the installation is wrong, or the code or data has been changed. Do not use the results until the cause is known.
5. Run `python run.py reproduce`. The file `outputs/baseline_2025/reproduce.csv` lists 17 values printed in Sections III–V next to the simulator's values. They must agree.
6. Reproduce the book's central scenario. Copy `docs/CALIBRATED_DIGEST.md` (the next command overwrites it), then run `python run.py calibrated --scenario calibrated_2026` (about 7 minutes), `python run.py design --scenario calibrated_2026` (about 20 minutes), `python run.py robustness --scenario calibrated_2026` (under a minute) and `python scripts/make_calibrated_digest.py calibrated_2026`. The new digest must match the published one, §§32–55, value for value. A difference means the installation or the code has changed (§8).

## 2. What the simulator computes

For one referendum it represents:

- **the electorate:** the register E, the residents able to vote R, and their ratio ρ = E/R;
- **the electorate's opinion:** the share μ of residents favouring change, which is unknown before the vote and drawn from a prior;
- **the campaigns:** two campaigns choose how many supporters to bring out above a sincere floor t_min that always votes;
- **the validity rule:** it decides whether the votes cast produce a valid change.

It then computes the campaigns' equilibrium. From the equilibrium it computes the criteria of Section I:

| Criterion | Meaning | Lower is better? |
|---|---|---|
| C1 | a majority of those who voted wanted change, but the rule kept the default | yes |
| C1′ | the rule produced change although most residents wanted the default | yes |
| C2 | what the default side gains by staying home (a boycott) | yes |
| C3 | the outcome differs from a vote on the question alone (government approval leaks in) | yes |
| C4 | an initiative with genuine support reaches a vote with a valid result | no: higher is better |
| C4′ | a signature list that is too short is admitted | yes |

Section VIII reads the criteria in three further ways (decision D14):

- **C1^R** is false invalidation against the residents' majority: the rule kept the default although most residents wanted change. **C1^Σ = C1′ + C1^R** (8.1) is the probability that the result goes against the residents' majority.
- **Binding effect.** An advisory (V9) or consultative (V11) Yes counts as no change, so those rules are scored on the change they make binding. The book's Pareto screen, **S3**, reads (C1^R, C1′, C2) on binding effect at printed precision (three decimals, ties kept).
- **The indifference weight w\*** (8.2). A drafter who counts a wrongly approved change as w times as bad as a wrongly blocked one scores a rule by w·C1′ + C1^R. Rule θ, with the lower C1′, is preferred to V1 if and only if w exceeds w\*(θ, V1); `viii1_TR22_weights.csv` gives w\* for every rule and cell (T1), `viii2_TR29_T6_accession_weights.csv` for T6 under accession priors, and `ix_TR30*.csv` across the one-at-a-time variants, for T3–T5 and local units, and under unequal stakes Ω (the argument `Omega` of `evaluate.cell`).

Turnout noise (A11) is added to the realised turnout of each side. The Monte Carlo layer checks the exact computation by simulating vote counts and applying the legal test to them.

## 3. Configuration

A scenario is a YAML file in `config/`. Each parameter carries its value, the source and the date:

```yaml
electorate:
  E_full:   {value: 3713761, source: "OSCE/ODIHR final report, 2025", date: "2025-10-23"}
  R:        {value: 1920000, range: [1890000, 1950000], source: "Section II", date: "2025-01-01"}
behaviour:
  tau_central: {value: 0.30, source: "D7 calibration point", date: "2026-10-09"}
  t0_resident: {value: null, source: "optional: hold resident turnout fixed", date: "2026-10-12"}
```

The groups are `electorate`, `rules_in_force`, `behaviour` (turnout, floor, polarisation, noise, prior), `drift`, `window`, `opinion`, `pipeline`, `signatures`, `contamination`, `campaign` and `monte_carlo`.

**Which scenario.** Since Section VII, Part 2 the book's central scenario is `calibrated_2026` (`config/calibrated_2026.yaml`); `baseline_2025` is kept for comparison with Sections VI and VII, Part 1. A rerun on new data starts from `calibrated_2026` (§4).

**Parameters without an update flag.** Edit them in the scenario file, each with its value, source and date (Appendix H, Table H.2, lists their sources and frequencies):

| Parameter | YAML key |
|---|---|
| Expected stand-alone turnout τ̂ | `behaviour.tau_central` (grid `behaviour.tau_grid`) |
| Contested, boycott and co-scheduled floors | `behaviour.t_min_central`, `behaviour.t_min_boycott`, `behaviour.t_min_co` |
| Reach t̄ | `behaviour.t_bar` |
| Shape of opinion a_B | `behaviour.a_B_central` (grid `behaviour.a_B_grid`) |
| Bloc alignment π and agenda setter μ_b by type | `behaviour.pi_by_type`, `behaviour.mu_b_by_type` |
| Unit dispersion σ_u | `behaviour.sigma_u_qark`, `behaviour.sigma_u_municipal` |
| Cue-taking share γ_c | `behaviour.gamma_c` |
| Window decay T_post, T_pre | `window.T_post_months`, `window.T_pre_months` |
| Persistence of opinion T_rel; same-issue threshold q_χ | `opinion.T_rel_years`, `opinion.q_chi` |
| Contamination Ψ and government approval μ_G | `contamination.Psi`, `contamination.Psi_by_type`, `contamination.mu_G` |
| Register volatility σ_lnρ | `drift.sigma_lnrho` |
| Valid share of signature lines v_L | `signatures.v_L_central` |
| Learning rate r̄_I and β_I | `campaign.r_bar_I`, `campaign.beta_I` |

Two switches change the model, not a value: `behaviour.a36` (a default side under a participation-type rule may call to abstain at the boycott floor; on in calibrated_2026) and `behaviour.noise_scaled_on_R` (on a resident denominator the turnout noise is σ_τ·ρ, since σ_τ is a share of E_full; on in calibrated_2026, off in baseline_2025).

To run a scenario: `python run.py packages --scenario <file name without .yaml>`. Results go to `outputs/<scenario name>/`.

Two files define what is compared:

- `config/types.yaml` holds the eight referendum types: the initiator, the default outcome, the rule in force, the denominator and the pipeline.
- `config/packages.yaml` holds the reform packages. Each package gives a rule for each type, a timing scenario (TS0–TS9), a window anchor, a verification rule, a mooting rule and the time allowed for T4 review. It also lists which of its parts need a constitutional amendment.

## 4. How to update the data

Use `scripts/update_data.py`. It copies a base scenario, replaces the figures you give, and stamps each one with your source and date. Give `--base calibrated_2026`, the book's central scenario; without it the script copies `baseline_2025`.

```bash
python scripts/update_data.py --base calibrated_2026 --name lists_2027 \
    --E-full 3780000 --E-ocv 262000 --R 1890000 \
    --source "KQZ final voter lists, 16 May 2027; INSTAT residents 18+ on 1 January 2027" --date 2027-05-16
python run.py calibrated --scenario lists_2027      # TR11-TR20, digest §§32-43, about 7 minutes
python run.py design --scenario lists_2027          # TR21-TR29, digest §§44-52, about 20 minutes
python run.py robustness --scenario lists_2027      # TR30-TR32, digest §§53-55, under a minute
python run.py pipeline --scenario lists_2027        # Pr_vote within 12/24/48 months by TS0-TS9 and President
python scripts/make_calibrated_digest.py lists_2027 # writes docs/CALIBRATED_DIGEST.md: copy the published one first
```

Which figures to update, and where to find them:

| Figure | Flag | Source | When |
|---|---|---|---|
| Registered voters E_full and out-of-country registrants E_ocv | `--E-full`, `--E-ocv` | KQZ final lists (OSCE/ODIHR report) | every election |
| Resident adults R | `--R` | INSTAT, residents 18+ on 1 January | every year |
| Annual growth of E | `--gamma-E` | KQZ lists, successive elections | every election |
| Adult decline rate | `--d-R-central` | INSTAT age structure | every year |
| Turnout volatility σ_τ | `--sigma-tau` | stand-alone votes, when any are held | when available |
| Municipal table | `--municipal file.csv` | KQZ lists by municipality; INSTAT residents by municipality | every census or election |
| A new Assembly cycle | `--election polling,end,first_sitting` | Official Journal; Assembly records | every election |

The municipal table must keep the column layout of `data/table_c4_municipalities.csv`. Two maps are shipped (v1.1.5): the 46 municipalities of Law No. 100/2026, from the 2027 local elections (`data/table_c4_municipalities_2027.csv`, the default), and the 61 of Law No. 115/2014 (`--map 2014`, outputs in `outputs/<scenario>_map2014/`). The resident column is rebased automatically to the R of the scenario (decision D11).

**Which run gives what.** Each run writes CSV files to `outputs/<scenario>/`, each with a `.meta.json` sidecar:

| Runs | Command | Digest | Files |
|---|---|---|---|
| TR11 packages at the centre and 15 variants | `run.py calibrated` | §§32–34 | `vii2_TR11_packages_central.csv`, `vii2_TR11_variants.csv` |
| TR12 60 common draws | `run.py calibrated` | §35 | `vii2_TR12_draws.csv`, `vii2_TR12_paired.csv`, `vii2_TR12_summary.csv` |
| TR13 lock boundary of Regime A | `run.py calibrated` | §36 | `vii2_TR13_lock_boundary.csv`, `vii2_TR13_aB_x_pi.csv` |
| TR14 T5 opposition thresholds | `run.py calibrated` | §37 | `vii2_TR14_T5_thresholds.csv` |
| TR15 complexity bound | `run.py calibrated` | §38 | `vii2_TR15_complexity.csv` |
| TR16 contamination C3 | `run.py calibrated` | §39 | `vii2_TR16_contamination.csv` |
| TR17 drift and first passage | `run.py calibrated` | §40 | `vii2_TR17_drift_lock.csv`, `vii2_TR17_first_passage.csv`, `vii2_TR17_ig.csv` |
| TR18 local units | `run.py calibrated` | §41 | `vii2_TR18_local_units.csv`, `vii2_TR18_local_summary.csv`, `vii2_TR18_P4_sigma_u.csv` |
| TR19 verification | `run.py calibrated` | §42 | `vii2_TR19_verification.csv` |
| TR20 the menu (T1) | `run.py calibrated` | §43 | `vii2_TR20_menu.csv`, `vii2_TR20_pareto.csv` |
| TR21 validity with procedure | `run.py design` | §44 | `viii1_TR21_central.csv`, `viii1_TR21_draws.csv`, `viii1_TR21_paired.csv`, `viii1_TR21_summary.csv` |
| TR22 binding screen and w* | `run.py design` | §45 | `viii1_TR22_pareto.csv`, `viii1_TR22_weights.csv` |
| TR23 T3, T4, T6 menus; accession priors | `run.py design` | §46 | `viii1_TR23_menu_T3_T4_T6.csv`, `viii1_TR23_pareto.csv`, `viii1_TR23_T6_accession.csv` |
| TR24 T5 menu | `run.py design` | §47 | `viii1_TR24_T5_central.csv`, `viii1_TR24_T5_variants.csv`, `viii1_TR24_T5_draws.csv`, `viii1_TR24_T5_summary.csv` |
| TR25 local menu | `run.py design` | §48 | `viii1_TR25_local_units.csv`, `viii1_TR25_local_summary.csv` |
| TR26 window | `run.py design` | §49 | `viii1_TR26_window_grid.csv`, `viii1_TR26_window_years.csv` |
| TR27 level boundaries | `run.py design` | §50 | `viii1_TR27_boundaries.csv`, `viii1_TR27_grid.csv` |
| TR28 complexity under V1 and V7 | `run.py design` | §51 | `viii1_TR28_complexity.csv` |
| TR29 w* for T6 under accession priors | `run.py design` or `run.py accession --means <m>` | §52 | `viii2_TR29_T6_accession_weights.csv` |
| TR30 V1/V7 boundary: variants, T3-T5, local units, unequal stakes | `run.py robustness` | §53 | `ix_TR30a_T1_variants.csv`, `ix_TR30b_types_units.csv`, `ix_TR30b_units_summary.csv`, `ix_TR30c_unequal_stakes.csv` |
| TR31 voting from abroad: 91-day lead; V1 with votes from abroad | `run.py robustness` | §54 | `ix_TR31a_TS9_lead91.csv`, `ix_TR31b_abroad.csv` |
| TR32 first-year cost of Pv61 (2026 initiatives) | `run.py robustness` | §55 | `ix_TR32_first_year.csv`, `ix_TR32_paths.csv` |
| Pr_vote by timing scenario | `run.py pipeline` | (F76) | `pipeline_Pr_vote.csv` |


## 5. How to add a rule or a type

**A new validity rule.** Open `refsim/rules/__init__.py` and copy the class of the nearest existing rule. A rule needs three methods:

1. `intervals(sx, sd, el)`: the support levels μ at which change wins, given the turnout rates of the two sides. Build it from the primitives `iv_majority`, `iv_approval`, `iv_participation`, `iv_supermajority` and `inter`.
2. `default_valid(sx, sd, el)`: the support levels at which the default wins *with a valid result* (needed for C4).
3. `evaluate(votes)`: the same legal test applied to vote counts.

Set `default_monotone = False` if adding ballots for the default can ever turn the default into change (Lemma 3). Register the class in `RULES`. Then add a test to `tests/test_rules.py`; `test_interval_view_matches_counts` checks that the two views agree.

**A new type.** Add an entry to `config/types.yaml` (default, initiator, rule in force, denominator, chain) and give it a rule in each package of `config/packages.yaml`.

**A new package.** Copy a package in `config/packages.yaml`, rename it, and change the rules or the timing. Add its name to `PACKAGES` in `refsim/runs.py`.

## 6. How to read regime maps

A regime map (Figure 6.1, `regime_map_V3.csv`) shows, for each register ratio ρ and stand-alone turnout τ, what the campaigns do in equilibrium:

- **Lock (gray):** nobody campaigns, only the floor votes, and the default wins with certainty, whatever residents think. This is the worst cell for C1 and C4.
- **Default demobilises (blue):** the default side stays at its floor because the quorum will defend it. The challenger campaigns, at full reach (dark blue) or below it (light blue). Change needs support μ̂ above one half, which is where C1 comes from.
- **Slack or contested (green):** both sides campaign, and the outcome is close to a simple majority.
- **No pure equilibrium:** the campaigns would keep changing their strategies (Section III's gap region Z2). The simulator reports an approximate mixed equilibrium and flags it.

The boundaries move with the prior, the floor and the register. Figure 6.2 and `prior_maps.csv` give the lock boundary for each prior shape. Figure 6.4 and `drift_lock_years.csv` give the year the register crosses each lock. `regime_map_pareto.csv` names, for each cell, the rules no other rule beats on C1, C1′ and C2 together.

A finding holds in a region of the map, not at a point. Report it with its boundary: "Regime A is locked for stand-alone turnout below about 0.30 of E under a uniform prior (0.22–0.46 across prior shapes)".

## 7. Worked example: a drafter reruns the analysis in 2027

*The figures below are an illustrative projection, not observed data. This example uses `baseline_2025` and the packages run of Section VI; for the book's central scenario, build the projection with `--base calibrated_2026` and run the commands of §4.*

Suppose the 2027 lists show 3,776,161 registered voters, of whom 258,000 are out-of-country registrants, and INSTAT reports 1,887,600 resident adults. The drafter wants to know whether current law still works, assuming residents vote as often as before.

1. **Create the scenario.**
   ```bash
   python scripts/update_data.py --name example_2027_projection --E-full 3776161 --E-ocv 258000 --R 1887600 \
       --source "ILLUSTRATIVE projection on the A29 path; not observed data" --date 2027-05-16
   ```
2. **Hold resident behaviour fixed.** In `config/example_2027_projection.yaml`, set `behaviour.t0_resident` to 0.5803, the 2025 resident turnout (0.30 × 1.9343). Turnout as a share of the register then falls as the register grows.
3. **Run:** `python run.py packages --scenario example_2027_projection --draws 10`.
4. **Read** `outputs/example_2027_projection/packages_central.csv` against the baseline:

| | ρ | Regime A (P0), T1 | Resident-indexed (P1), T1 |
|---|---|---|---|
| 2025 baseline | 1.934 | default demobilises; Pr[change] 0.277; C1 0.546 | Pr[change] 0.544; C1 0.249 |
| 2027 projection | 2.000 | **lock**; Pr[change] 0; C1 0.500; C4 0 | Pr[change] 0.543; C1 0.250 |

At unchanged resident behaviour, the growth of the register alone moves Regime A into the lock: no initiative can pass, whatever residents think. The resident-indexed rule is unaffected. This is F45 and F65 of the book, reproduced on new data. The review indicators (`refsim/review.py`, Table V.28) would have flagged it through I1 and I3.

## 8. Reproducibility and maintenance

- **Seeds** are fixed in the scenario (`monte_carlo.seed`) and in every Monte Carlo function.
- **Exact methods first.** Equilibria are solved on a grid and then polished by continuous best responses. Criteria use Gauss–Hermite quadrature (80 × 80 nodes) under noise. The Monte Carlo checks these to within four standard errors (`run.py convergence`).
- **Versioning.** Tag a release after each data update, and name output folders by scenario and vintage. The `.meta.json` sidecars record every input.
- **Legacy code.** `legacy/` holds the code archives exactly as they passed the independent checks of Appendices B1 and B2. The simulator imports the pipeline chain and the Jacobi solver from there. Do not edit them; if a correction is needed, add a wrapper and a test.
- **What is not yet wrapped.** These parts are in `legacy/` or outside the tool and cannot be rerun from it: the early-election Monte Carlo of Section V Part 1 (λ_ee > 0), the local timing chain (LT1–LT4), the station-level trajectory test of Section IV, the airtime and poll modules, and the two-component opinion correlation (7.9). See `docs/ASSUMPTIONS.md`.
- **TS9 as computed.** The pipeline follows one initiative at a time. Under TS9 polling falls within 26 weeks of the Court's ruling, at the earliest or latest lawful week (outside the windows, not an election day), or on the first lawful week after them if none lies within them. Two referendums sharing the one annual day (Section VIII's Pv17 carry-over) are not modelled.

## 9. Using this manual with Appendix H

Appendix H (the drafting checklist) says when to rerun (H.5), which figures to update and where they come from (Table H.2), which review trigger sends the drafter back to which step (H.12), and how to record each decision with its scenario, refsim version and `.meta.json` sidecar (Form H.1). Cite every number from a rerun as a refsim output: version, scenario, and CSV file or digest section.

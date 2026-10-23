# Assumptions made for missing parameters, and differences from Sections III–V

## A. Assumptions made because a parameter was missing or unsettled

| # | Where | Assumption | Status in the book |
|---|---|---|---|
| 1 | Central cell | τ = 0.30 of E_full (D7 calibration point), t_min = 0.20 (D9), π = 0, uniform prior (A2), σ_τ = 0.04, Θ = 0.5 (A11) | σ_τ [DATA NEEDED]: no stand-alone vote since 1998 |
| 2 | Noise | Strategies are those of the noiseless game; noise acts on realised turnout only (Section IV convention) | a noise-anticipating default side is [OPEN] |
| 3 | Criteria | C1, C1′, C2 follow the operational reading (3.6): joint probabilities; C2 is the fixed-opponent gain of stopping the default campaign (t_d = 0 against t_d = t₀ − t_min) | (3.6) [VERIFY against (I.7)–(I.8)] |
| 4 | Valid result (C4) | Approval-type rules (V3, V5a, V6, V8 approval limb, V9) are symmetric: the default also needs the bar for a valid result, as Art. 118(3) requires of "the winning alternative". V1, V4 and V7 always give a valid result. Under V9 the default is valid at the lower tier. | follows Section I, (I.10), and F52 |
| 5 | C4 by type | T1–T2: exact pipeline (Section V Part 1, B.43) × Pr_dec, reported both against an opposed majority (Pr_opp = 1, no mooting rule moots every initiative that would pass) and a friendly one (Pr_opp = 0). T3, T6: Pr_dec. T4: share of adoption weekdays with a feasible Sunday (Proposition 23) × Pr_dec, with review-plus-decree of 60 days under current law (C4 = 0) and 24 days under P5. T5: Pr_dec. | Remark 4, Proposition 23; the 60-day reading is the legal maximum, not observed practice |
| 6 | KQZ stage | False rejection at KQZ is the verification rule's error for a national list with v_L = 0.85 and a 5% margin | Section V Part 1 baseline |
| 7 | V10 | The equilibrium uses homogeneous units (the unit limb then coincides with the national majority). Unit heterogeneity enters only the Monte Carlo: μ_u = μ + 0.05·ε_u, qark weights Ê_q (2025 basis). | σ_u = 0.05 illustrative [DATA NEEDED: municipal Yes-share dispersion] |
| 8 | V9 and V11 | λ_adv = 0.5 (value of an advisory result relative to a binding one); V9 tiers 20% and one third of E | illustrative (Lemmas 6–7) |
| 9 | V5 | T_ref = in-country ballots at the last Assembly election, τ_R^ref = 0.76 | D2 |
| 10 | P3 (April 2026 draft) | "Repeal referendums valid at 40% of registered voters" read as a participation quorum (V2); the approval reading is P3a. Constitutional "above 50%" read as 50% for T4; T5 keeps Art. 125(4). "Consultative referendums" modelled as V11 (λ = 0.5) for T3. T2 and T6–T8 kept as in force. | [VERIFY against the draft's text] |
| 11 | P4 | Majority in 7 of 12 qarks | k_U of m_U is a design variable (D6) |
| 12 | Local types | Units at fixed resident turnout t₀ = 0.580 (the national value at τ = 0.30); R̂_u rebased to R = 1.92M (D11); unit denominators (D4) | F46 uses τ = 0.30 nationally |
| 13 | Uncertainty intervals | 60 draws per package: R ~ U[1.89, 1.95]M, τ ~ U[0.20, 0.50], t_min ∈ {0, 0.20, 0.35} equally likely, π ~ U[0, 0.3]. Percentiles from 60 draws carry their own sampling error (about ±0.03 at the 2.5% and 97.5% points). | the book's ranges D3 and D9 |
| 14 | Drift | A29 deterministic path from April 2025 lists; d_R = 0.0085 central, 0.0116 fast, 0.004 slow; lock year = calendar year of crossing | D11 |
| 15 | Calendar | Assembly cycles after 2025 follow A27 (polling on the second Sunday of May; t_end = previous first sitting + 4 years; t₁ = t_end + 2 days); local elections every four years from 2027 | [VERIFY] dates from 2029 |
| 16 | T6 | Assumed rule V3 with "not ratified" as default (D5) | no rule in force (F10, F16) |
| 17 | Mixed equilibria | Where no pure equilibrium exists (40 of 1,596 grid cells, all at lower bars: Z2), an approximate mixed equilibrium is computed by fictitious play on a 121-point grid and flagged; its exploitability is stored | Proposition 1(ii): existence only; mixed equilibria [OPEN] |
| 18 | Prior families | Beta(a_B, a_B) with a_B ∈ {0.5, …, 16}; zealot family at π > 0 uses f_μ(1/2) for κ | zealot κ at π > 0 [OPEN] |

## B. Numbers that differ from the printed text, and why

| Item | Printed | Simulator | Reason |
|---|---|---|---|
| ρ in the central cell | 1.93 / 1.81 (Sections III–IV) | 1.9343 / 1.8062 | D10: exact E/R; tables printed at rounded ρ are tested at rounded ρ |
| μ̂_A on the capped branch; Pr[change] at π = 0 | 0.643 / 0.603; 0.357 / 0.397 | 0.645 / 0.602; 0.355 / 0.398 | D10 |
| Municipal map | 46 municipalities of Law No. 100/2026 (the book printed the 61 of Law No. 115/2014 until October 2026) | 46 (Law No. 100/2026, adopted 9 October 2026, from the 2027 local elections), the default from v1.1.5; `--map 2014` gives the 61 | Table C.4 aggregated by whole-municipality mergers; the 12 qarqe unchanged |
| Municipalities with q_A·ρ̂_u > 1 | 6 (46-unit map; 8 on the 61-unit map) | 7 (2025 basis; Pustec 3.011; 9 on the 61-unit map) | D11: rebasing of Table C.4a |
| Remark 1 exceptions | 2 (Tiranë, Kamëz; 4 on the 61-unit map, with Klos and Dimal) | 2 (3 on the 61-unit map: Dimal, at 1.503, has left) | D11 |
| Lock years at fixed resident turnout | 2027, 2031, 2034, 2036, 2038, 2040 | 2027, 2032, 2036, 2038, 2040, 2043 (central); the printed years are the fast case | D11: adult decline 0.85% a year |
| Municipal counts under drift | 10 (2027), 13 (2030) (61-unit map: 12, 16) | 10, 12 on a 2025 basis at the adult rate (10, 14 in the fast case; 61-unit map: 12, 16 and 13, 18) | D11 |
| T7 units locked at the central cell | 33–37 of 46 (F46, τ = 0.30 nationally; 45–49 of 61) | 33 of 46 at t₀ = 0.580 (45 of 61); 44 of 46 calibrated (57 of 61) | Assumption 12 |
| Feasible Sundays, 2061 | legacy calendar stops at its 2057 cycle | extended by A27 | horizon only |

## C. Gaps the simulator does not yet fill

1. The early-election calendar (λ_ee > 0) and the current-mandate calendar of Section V Part 1. They are available in `legacy/section5_part1_B2/pipeline.py` (`mc_nf`) but not wrapped; C4 is reported for regular terms only.
2. The local timing chain (LT1–LT4) for T7–T8: local C4 is Pr_dec only.
3. The station-level trajectory test (Lemma 13) and the airtime and poll modules of Section III Part 2 (Propositions 10–11). The poll model is in `legacy/section4_B1/s4_core.py` (`poll_model`).
4. Contamination C3 needs Ψ and μ_G from Section VII. With the default Ψ = 0, C3 = 0. Runs with Ψ > 0 are available through `evaluate.cell(..., Psi=, mu_Gx=)`.
5. Calibration of T_rel, a_B, T_post, σ_τ and t_min (Section VII). Every one of them is a scenario parameter.

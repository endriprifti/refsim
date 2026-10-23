# Proposition 30 — Regime B (T5) under general priors

*Draft for Section VI (to be placed in Appendix B as B.55). It settles the qualifier on F28 raised by Consistency check 2 (Check 8, block R9). Code: `refsim/regime_b.py`; tests: `tests/test_section_V.py::test_prop30_*`.*

## Setting

Assumptions A1 (τ_ab = 0) and A12, with π = 0. A T5 referendum is decided under Regime B (V4, q_R = 1/2). The amendment falls only if the No votes exceed half of E. In resident units the bar is Q_R = ρ/2 ≥ 1/2 for ρ ≥ 1. By Lemma 4(a) the reform side stays at its floor t_min. The T5 opposition (the challenger, x) chooses a rate s ∈ [t_min, t̄] with t̄ = 1. Its payoff, in units of its stake B, is

  u_F(s) = S_F(Q_R/s) − f_cal · (s − t_min)² / (8 t0 (t0 − t_min)),  (P30.1)

where S_F is the survival function of the prior F on [0, 1] and f_cal = f_F(1/2) (κ recalibrated, A12) or f_cal = 1 (κ held at the uniform calibration). Because Q_R ≥ 1/2 > t_min for every t_min in D9, Q_R/t_min > 1, so the floor payoff is S_F(Q_R/t_min) = 0. The opposition campaigns iff its gain G_F = max_{s > t_min} u_F(s) is positive. Under A2 (F = U, uniform) this is the condition of Proposition 19 and (3.11)–(3.11′).

## Proposition 30

**(i)** Let F be symmetric about 1/2 with a density f that is non-increasing on [1/2, 1] (symmetric unimodal). Then for every s ∈ (t_min, t̄], u_F(s)/f_cal ≤ u_U(s) under either κ convention. Hence G_U ≤ 0 implies G_F ≤ 0. Wherever F28's clause holds under A2 (no T5 campaign within D3), it holds under every symmetric unimodal prior.

**(ii)** For U-shaped priors the clause can fail. Under Beta(a, a) with a < 1, the opposition campaigns at the cap iff t0 ≥ t0*, where

  t0* = ½ [ t_min + √( t_min² + (t̄ − t_min)² / (2K) ) ],  K = S_F(Q_R/t̄)/f_cal,  (P30.2)

that is, at registered turnout τ ≥ t0*/ρ. At the 2025 register with κ recalibrated:

| prior | E_dom (ρ = 1.806): t_min = 0 / 0.20 / 0.35 | E_full (ρ = 1.934): t_min = 0 / 0.20 / 0.35 |
|---|---|---|
| a = 0.5 | 0.348 / 0.339 / 0.343 | 0.428 / 0.398 / 0.383 |
| a = 0.75 | 0.481 / 0.444 / 0.424 | none within D3 |
| a ≥ 1 | none within D3 | none within D3 |

The general best response over every rate s gives the same thresholds in every cell computed: the cap is the opposition's best campaigning rate. That last statement is *computed*, not proved.

## Proof of (i)

1. **Range of the threshold.** For s ∈ (t_min, t̄], y ≡ Q_R/s ≥ Q_R ≥ 1/2. If y ≥ 1, then S_F(y) = S_U(y) = 0 and the claim is trivial. Suppose y ∈ [1/2, 1).
2. **κ recalibrated.** Since f is non-increasing on [1/2, 1], S_F(y) = ∫_y^1 f(m) dm ≤ (1 − y) f(y) ≤ (1 − y) f(1/2). Dividing by f_cal = f(1/2) gives S_F(y)/f_cal ≤ 1 − y = S_U(y), because f_U ≡ 1. In (P30.1) divided by f_cal, the cost term (s − t_min)²/(8t0(t0 − t_min)) is the same for F and U. So u_F(s)/f_cal ≤ u_U(s).
3. **κ fixed.** Here f_cal = 1 for both priors. By symmetry ∫_{1/2}^1 f = 1/2, so the mean of f over [1/2, 1] equals 1. A non-increasing function has a mean over any right sub-interval [y, 1] no larger than its mean over [1/2, 1]. Hence S_F(y) = (1 − y)·mean_{[y,1]} f ≤ 1 − y = S_U(y), and u_F(s) ≤ u_U(s).
4. **Conclusion.** Both floor payoffs are zero (Setting). So G_F ≤ f_cal·G_U under recalibration and G_F ≤ G_U under the fixed convention. If G_U ≤ 0, then G_F ≤ 0. ∎

*Remark.* The argument is that of Corollary 5 (Regime A), applied to the rejection bar: among symmetric unimodal priors, the uniform puts the most mass in the far tail relative to its density at 1/2. The same steps apply for π > 0 with the uniform on the support [π/2, 1 − π/2] as the comparison.

## Proof of (ii)

At s = t̄ the payoff is f_cal·[K − (t̄ − t_min)²/(8 t0 (t0 − t_min))]. It is positive iff t0(t0 − t_min) ≥ (t̄ − t_min)²/(8K). The positive root of this quadratic in t0 is (P30.2). For Beta(a, a) with a < 1 the density is U-shaped, so f(1/2) is its minimum and K can exceed 1 − Q_R/t̄. For a = 0.5 on E_dom, K = S(0.903)/f(1/2) = 0.2015/0.6366 = 0.3165, against 0.097 under A2. The numbers in the table come from (P30.2) and from the general best response (`regime_b.tau_threshold_solver`). The two agree to the bisection tolerance. ∎

## Consequence for the text

F28 and F35 hold "under A2 or any symmetric unimodal prior". Under polarised opinion the T5 request is no longer delay-only once stand-alone turnout reaches about 0.34 of E_dom (0.38–0.43 of E_full). Section VII should test the shape of the prior from cross-question support (a_B < 1 or not) before the "delay only" reading is relied on in Section VIII.

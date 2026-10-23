"""Section VII case runs (refsim v1.0.3): tool runs R1-R10 requested by Section VII, Part 1.

Every run uses the engine of Sections III-VI unchanged (game.Game, rules, priors, pipeline, signatures);
the only additions are the combined-limb rule V8x (rules) and the A5 co-scheduled game (GameCo, below).
Outputs: outputs/<scenario>/vii_R*.csv with .meta.json; the digest is docs/CASES_DIGEST.md
(scripts/make_cases_digest.py).

Conventions (stated in each CSV's note):
- Historical cases (R1-R3, R9) use the observed turnout of the vote as the stand-alone baseline tau of the
  model.  Observed turnout is itself an equilibrium outcome, so these runs place a case in a regime; they
  are not forecasts of its turnout.
- Resident adults for 1994-1998 and for the comparators are unknown: rho is run on a grid (tau_ab = 0).
- Beta priors are run at pi = 0 only (A12 is defined there); A2 is run at every pi with the stated mu_b.
- Turnout noise A11: sigma_tau = 0.04, Theta = 0.5; noiseless values are reported alongside.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

from . import pipeline, report, signatures
from .electorate import Electorate, load_municipalities, national
from .game import Game, noise_nodes, tau_lock
from .priors import Prior, uniform
from .rules import V1, V2, V3, V7, V8, V8x, V10, V11, Votes, make_rule

SIG, THETA = 0.04, 0.5


# ---------------------------------------------------------------- helpers
def prior_of(family, pi=0.0, mu_b=0.5, a_B=1.0):
    if family == "A2":
        return uniform(pi, mu_b)
    return Prior(a_B, a_B, 0.0, 1.0, label=f"Beta({a_B:g},{a_B:g})")


def bar_ratio(rule, el, t0):
    """Lambda = Q/t0 for the rule's binding E-indexed limb (participation or approval), else nan."""
    if isinstance(rule, V2):
        return el.Q_participation(rule.params["q"]) / t0
    if isinstance(rule, V3):
        return el.Q_approval(rule.params["q"]) / t0
    if isinstance(rule, V8x):
        qs = [el.Q_participation(rule.q_P)] if rule.q_P is not None else []
        qs += [el.Q_approval(rule.q_A)] if rule.q_A is not None else []
        return max(qs) / t0 if qs else float("nan")
    if isinstance(rule, V8) and rule.params.get("limb") == "participation":
        return el.Q_participation(rule.params["q"]) / t0
    if isinstance(rule, V11):
        return bar_ratio(rule.base, el, t0)
    return float("nan")


def solve_row(game: Game, sigma=SIG):
    eq = game.solve()
    c = game.criteria(eq, sigma_tau=sigma, Theta=THETA)
    c0 = game.criteria(eq, sigma_tau=0.0, Theta=THETA)
    tol = 1e-6
    d_floor = abs(eq.sd - game.t_min) < tol
    x_floor = abs(eq.sx - game.t_min) < tol
    return eq, dict(regime=c["regime"], sx=eq.sx, sd=eq.sd, pure=eq.pure, exploitability=eq.exploitability,
                    default_side=("abstains (at its floor)" if d_floor and not x_floor else
                                  ("both at floor (lock)" if d_floor and x_floor else "campaigns")),
                    Lambda=bar_ratio(game.rule, game.el, game.t0), t0=game.t0,
                    mu_hat=c["mu_hat_eq"], P_change=c["P_change"], P_change_noiseless=c0["P_change"],
                    P_dec=c["P_dec"], C1=c["C1"], C1p=c["C1p"], C1R=c["C1R"], C2=c["C2"], C2_noiseless=c0["C2"],
                    turnout_E=c["turnout_E"], turnout_E_noiseless=c0["turnout_E"])


def pr_change_given_mu(game: Game, eq, mu: float, sigma=SIG):
    """Pr[change | mu] at the equilibrium, under A11 noise (Gauss-Hermite nodes)."""
    pairs = [(eq.sx, eq.sd, 1.0)] if eq.pure else [
        (eq.grid[i], eq.grid[j], eq.mix_x[i] * eq.mix_d[j]) for i in range(len(eq.grid)) for j in range(len(eq.grid))
        if eq.mix_x[i] * eq.mix_d[j] > 1e-4]
    tot = wsum = 0.0
    for sx, sd, w in pairs:
        X, D, W = noise_nodes(sx, sd, game.el.rho * sigma, THETA, game.t_bar)
        ins = np.zeros_like(W)
        for p in game.rule.intervals(X, D, game.el):
            if p.kind == "change":
                ins = np.maximum(ins, ((np.asarray(p.a) <= mu) & (mu <= np.asarray(p.b))).astype(float))
        tot += w * float(np.sum(ins * W))
        wsum += w
    return tot / wsum


def implied_mu(y_V, sx, sd):
    """Resident support that yields Yes share y_V of valid votes at side rates (sx, sd), tau_ab = 0."""
    den = sx * (1 - y_V) + y_V * sd
    return y_V * sd / den if den > 0 else float("nan")


# ---------------------------------------------------------------- A5 co-scheduled game
class GameCo(Game):
    """Proposition 12 under A5: the floor rises to t_min_co, kappa stays at the stand-alone calibration
    (cost denominator 8 t0_sa (t0_sa - t_min_sa)); t0 is the co-scheduled V1 rate,
    s(s - t_min_co) = t0_sa (t0_sa - t_min_sa) (capped at t_bar)."""

    def __init__(self, rule, el, t0_sa, t_min_sa, t_min_co, prior, pi=0.0, t_bar=1.0):
        k = t0_sa * (t0_sa - t_min_sa)
        s_v1 = min(t_bar, 0.5 * (t_min_co + math.sqrt(t_min_co ** 2 + 4 * k)))
        super().__init__(rule, el, t0=s_v1, t_min=t_min_co, prior=prior, pi=pi, t_bar=t_bar)
        self.k_sa = k

    def cost(self, s):
        return self.fcal * (np.asarray(s, float) - self.t_min) ** 2 / (8 * self.k_sa)


def pi_star_co(rule, el, t0_sa, t_min_sa, t_min_co, mu_b):
    def g(pi):
        return GameCo(rule, el, t0_sa, t_min_sa, t_min_co, uniform(pi, mu_b), pi).campaign_gain()
    if g(0.0) <= 0:
        return float("nan")
    if g(0.999) > 0:
        return 1.0
    return float(optimize.brentq(g, 0.0, 0.999, xtol=1e-7))


# ---------------------------------------------------------------- R1 1994
def run_R1(P, scen):
    E = 1_985_986
    rows = []
    priors = [("A2", 0.0, 0.5, 1.0)] + [("A2", pi, mb, 1.0) for pi in (0.3, 0.5) for mb in (0.5, 2 / 3)] + \
             [("Beta", 0.0, 0.5, a) for a in (0.5, 2.5)]
    for rule in (V2(q=0.5), V3(q=1 / 3)):
        for rho in (1.00, 1.05, 1.10, 1.15):
            el = Electorate(E=E, R=E / rho, label=f"1994 rho={rho:g}")
            for tau in (0.60, 0.70, 0.8443):
                for tm in (0.0, 0.20, 0.35):
                    for fam, pi, mb, a in priors:
                        g = Game(rule, el, tau=tau, t_min=tm, prior=prior_of(fam, pi, mb, a), pi=pi)
                        base = dict(case="1994", rule=rule.describe(), rho=rho, tau=tau, t_min=tm, pi=pi, mu_b=mb,
                                    prior=g.prior.label)
                        if not g.feasible:
                            rows.append(dict(base, feasible=False)); continue
                        eq, r = solve_row(g)
                        m = implied_mu(0.4362, eq.sx, eq.sd)
                        rows.append(dict(base, feasible=True, **r, mu_implied_by_yV=m,
                                         turnout_E_at_mu_implied=(m * eq.sx + (1 - m) * eq.sd) / rho))
    report.write_csv(rows, "vii_R1_1994", P, scen,
                     note="R1: 1994 constitutional referendum (T4 analogue; challenger = Yes). E = 1,985,986; R unknown "
                          "(rho grid); observed turnout 0.8443 used as the stand-alone baseline; V2 at 1/2 (the 1994 rule "
                          "as read in VII.1.3) and V3 at 1/3 (today's Regime A). mu_implied_by_yV: resident support "
                          "that gives the observed Yes share 0.4362 at the equilibrium rates; turnout_E_at_mu_implied: the "
                          "model's turnout there (observed 0.8443).")
    # the floor at which the 1994 vote becomes contested at its observed turnout (grid 0.005)
    fl = []
    for rule in (V2(q=0.5), V3(q=1 / 3)):
        for rho in (1.00, 1.05, 1.10, 1.15):
            el = Electorate(E=E, R=E / rho)
            for fam, pi, mb, a in priors:
                first = None
                for tm in np.round(np.arange(0.0, 0.60 + 1e-9, 0.005), 3):
                    g = Game(rule, el, tau=0.8443, t_min=tm, prior=prior_of(fam, pi, mb, a), pi=pi)
                    if not g.feasible:
                        continue
                    eq = g.solve()
                    if g.regime(eq).startswith("slack") or g.regime(eq) == "contested":
                        first = float(tm)
                        break
                fl.append(dict(rule=rule.describe(), rho=rho, pi=pi, mu_b=mb, prior=prior_of(fam, pi, mb, a).label,
                               t_min_contested_from=first, eps_contested_from=(first / (rho * 0.8443)) if first is not None else None))
    report.write_csv(fl, "vii_R1_1994_floor", P, scen,
                     note="R1: lowest sincere floor t_min (grid 0.005) at which the 1994 vote is contested (both sides campaign: "
                          "slack or contested) at the observed turnout 0.8443 taken as the stand-alone baseline; eps = t_min/t0.")
    return rows


# ---------------------------------------------------------------- R2 1997 (co-scheduled)
def run_R2(P, scen):
    E = 1_986_550
    mu_obs = 0.3325
    mu_b = 0.2895
    rows = []
    for rule in (V2(q=0.5), V3(q=1 / 3)):
        for rho in (1.00, 1.05, 1.10):
            el = Electorate(E=E, R=E / rho, label=f"1997 rho={rho:g}")
            for tau_sa in (0.30, 0.40, 0.50):
                t0_sa = rho * tau_sa
                tm_sa = 0.20
                gs = Game(rule, el, t0=t0_sa, t_min=tm_sa, prior=uniform(0.0), pi=0.0)
                _, rs = solve_row(gs)
                for t_co in (0.70, 0.72, 0.75):
                    for pi in (0.0, 0.3, 0.5):
                        pr = uniform(pi, mu_b)
                        g = GameCo(rule, el, t0_sa, tm_sa, t_co, pr, pi)
                        eq, r = solve_row(g)
                        rows.append(dict(case="1997", rule=rule.describe(), rho=rho, tau_standalone=tau_sa,
                                         t_min_standalone=tm_sa, t_min_co=t_co, pi=pi, mu_b=mu_b,
                                         pi_star_co=pi_star_co(rule, el, t0_sa, tm_sa, t_co, mu_b),
                                         mu_hat_standalone=rs["mu_hat"], regime_standalone=rs["regime"],
                                         Pr_change_at_mu_obs=pr_change_given_mu(g, eq, mu_obs),
                                         resident_turnout=r["turnout_E"] * rho, **r))
    report.write_csv(rows, "vii_R2_1997", P, scen,
                     note="R2: 1997 monarchy referendum co-held with the Assembly election (A5: floor t_min_co, kappa at "
                          "the stand-alone calibration with stand-alone tau and t_min = 0.20). Monarchy = change, "
                          "mu_b = 0.2895 (PD + Legaliteti list share). Pr_change_at_mu_obs at mu = 0.3325 (observed "
                          "Monarchy share of valid votes, a proxy for resident support).")
    return rows


# ---------------------------------------------------------------- R3 1998
def run_R3(P, scen):
    E = 1_914_859
    y_obs, tau_obs = 0.9351, 0.5057
    rows = []
    priors = [("A2", 0.0, 0.5, 1.0), ("A2", 0.3, 0.53, 1.0), ("A2", 0.3, 0.71, 1.0), ("Beta", 0.0, 0.5, 0.5)]
    for rule in (V2(q=0.5), V1(), V3(q=1 / 3)):
        for rho in (1.00, 1.05, 1.10):
            el = Electorate(E=E, R=E / rho, label=f"1998 rho={rho:g}")
            for tau in (0.45, 0.50, 0.5057, 0.55, 0.60):
                for tm in (0.07, 0.12, 0.20, 0.35):
                    for fam, pi, mb, a in priors:
                        g = Game(rule, el, tau=tau, t_min=tm, prior=prior_of(fam, pi, mb, a), pi=pi)
                        base = dict(case="1998", rule=rule.describe(), rho=rho, tau=tau, t_min=tm, pi=pi, mu_b=mb,
                                    prior=g.prior.label)
                        if not g.feasible:
                            rows.append(dict(base, feasible=False)); continue
                        eq, r = solve_row(g)
                        m = implied_mu(y_obs, eq.sx, eq.sd)
                        rows.append(dict(base, feasible=True, **r, mu_implied_by_yV=m,
                                         turnout_E_at_mu_implied=(m * eq.sx + (1 - m) * eq.sd) / rho))
    report.write_csv(rows, "vii_R3_1998", P, scen,
                     note="R3: 1998 constitutional referendum (T4 analogue). E = 1,914,859; rho grid; observed turnout used "
                          "as the stand-alone baseline. V2 at 1/2 = the repealed rule (counterfactual); V1 = the rule in "
                          "force; V3 at 1/3 = today's Regime A. mu_implied_by_yV: resident support that gives the observed "
                          "Yes share 0.9351 at the equilibrium rates; turnout_E_at_mu_implied: the model's turnout there "
                          "(observed 0.5057).")
    # tau above which V2 is slack (V1 outcome), and above which the default side campaigns
    sl = []
    for rho in (1.00, 1.05, 1.10):
        el = Electorate(E=E, R=E / rho)
        for tm in (0.07, 0.12, 0.20, 0.35):
            for fam, pi, mb, a in priors:
                pr = prior_of(fam, pi, mb, a)
                first_slack = first_campaign = None
                taus = np.round(np.arange(0.30, min(1.0 / rho, 0.95) + 1e-9, 0.005), 3)
                for tau in taus:
                    g = Game(V2(q=0.5), el, tau=tau, t_min=tm, prior=pr, pi=pi)
                    if not g.feasible:
                        continue
                    eq = g.solve()
                    reg = g.regime(eq)
                    if first_campaign is None and eq.pure and abs(eq.sd - tm) > 1e-6:
                        first_campaign = float(tau)
                    if first_slack is None and reg.startswith("slack"):
                        first_slack = float(tau)
                    if first_slack is not None and first_campaign is not None:
                        break
                sl.append(dict(rho=rho, t_min=tm, pi=pi, mu_b=mb, prior=pr.label,
                               tau_default_side_campaigns_from=first_campaign, tau_V2_slack_from=first_slack))
    report.write_csv(sl, "vii_R3_1998_V2_slack", P, scen,
                     note="R3: lowest stand-alone turnout (share of E, grid 0.005) at which, under V2 at 1/2, the default "
                          "side campaigns (no boycott) and at which V2 is slack (the V1 outcome).")
    return rows


# ---------------------------------------------------------------- R4 Italian analogue
def _pooled_stats(A):
    T, yV, val = A[:, 0], A[:, 1], A[:, 2].astype(bool)
    f = ~val

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    return dict(n_pooled=len(T), share_void=float(f.mean()),
                share_void_with_Yes_majority=float((yV[f] > 0.5).mean()) if f.any() else float("nan"),
                median_yV_void=float(np.median(yV[f])) if f.any() else float("nan"),
                r_void=r(T[f], yV[f]), r_valid=r(T[val], yV[val]), r_pooled=r(T, yV))


def run_R4(P, scen, n=6000, seed=7):
    import csv as _csv
    import os
    from .params import DATA_DIR
    it = list(_csv.DictReader(open(os.path.join(DATA_DIR, "cases", "italy_abrogative_1974_2025.csv"), encoding="utf-8")))
    obs_t = np.array([float(r["turnout"]) for r in it]) / 100
    grid = np.round(np.arange(0.20, 0.90 + 1e-9, 0.05), 2)
    idx = np.clip(np.round((obs_t - 0.20) / 0.05).astype(int), 0, len(grid) - 1)
    wts = np.bincount(idx, minlength=len(grid)) / len(obs_t)
    rng = np.random.default_rng(seed)
    cells, summ = [], []
    keep = {}
    for rho in (1.00, 1.05, 1.11):
        for tm in (0.05, 0.10, 0.20, 0.35):   # 0.35 added for contested high-turnout votes
            for fam, a in (("A2", 1.0), ("Beta", 0.5), ("Beta", 2.5)):
                pr = prior_of(fam, 0.0, 0.5, a)
                pools = {"Italian observed-turnout weights": [], "campaign cells, equal weights": [],
                         "boycott cells, equal weights": [], "contested or slack cells, equal weights": []}
                for k, tau in enumerate(grid):
                    t0 = rho * tau / (1 + 0.78 * (rho - 1))
                    el = Electorate(E=rho, R=1.0, tau_ab=0.78 * t0, label="Italy analogue")
                    g = Game(V2(q=0.5), el, t0=t0, t_min=tm, prior=pr, pi=0.0)
                    if not g.feasible:
                        continue
                    eq, r = solve_row(g)
                    mu = pr.sample(rng, n)
                    if eq.pure:
                        sx = np.full(n, eq.sx); sd = np.full(n, eq.sd)
                    else:
                        sx = rng.choice(eq.grid, n, p=eq.mix_x); sd = rng.choice(eq.grid, n, p=eq.mix_d)
                    sig = el.rho * SIG
                    Z1 = rng.standard_normal(n)
                    Z2 = THETA * Z1 + math.sqrt(1 - THETA ** 2) * rng.standard_normal(n)
                    X = np.where(sx > 0, np.clip(sx + sig * Z1, 0, 1), 0.0)
                    D = np.where(sd > 0, np.clip(sd + sig * Z2, 0, 1), 0.0)
                    Y = mu * X + el.mu_A * el.n_ab
                    N = (1 - mu) * D + (1 - el.mu_A) * el.n_ab
                    ok = (Y + N) > 0          # draws with no ballots (both rates clipped to 0) carry no Yes share
                    mu, Y, N = mu[ok], Y[ok], N[ok]
                    T = (Y + N) / el.rho
                    yV = Y / (Y + N)
                    valid = T >= 0.5
                    campaign = eq.sx > tm + 1e-6 or not eq.pure
                    boycott = eq.pure and abs(eq.sd - tm) < 1e-6 and eq.sx > tm + 1e-6
                    bins = np.quantile(mu, np.linspace(0, 1, 21))
                    cc = []
                    for b in range(20):
                        s_ = (mu >= bins[b]) & (mu <= bins[b + 1])
                        if s_.sum() > 30 and np.std(T[s_]) > 0 and np.std(yV[s_]) > 0:
                            cc.append(np.corrcoef(T[s_], yV[s_])[0, 1])
                    cells.append(dict(rho=rho, t_min=tm, prior=pr.label, tau=tau, weight=wts[k], regime=r["regime"],
                                      default_side=r["default_side"], sx=eq.sx, sd=eq.sd, Pr_valid=float(valid.mean()),
                                      r_tau_yV_across_prior=float(np.corrcoef(T, yV)[0, 1]) if np.std(T) > 0 else float("nan"),
                                      r_tau_yV_given_mu=float(np.mean(cc)) if cc else float("nan"),
                                      median_yV_void=float(np.median(yV[~valid])) if (~valid).any() else float("nan"),
                                      C1=r["C1"], C2=r["C2"], P_change=r["P_change"]))
                    blk = np.column_stack([T, yV, valid])
                    if wts[k] > 0:
                        m = max(1, int(round(wts[k] * 20000)))
                        pools["Italian observed-turnout weights"].append(blk[rng.choice(len(T), m)])
                    if campaign:
                        pools["campaign cells, equal weights"].append(blk[rng.choice(len(T), 1500)])
                    if boycott:
                        pools["boycott cells, equal weights"].append(blk[rng.choice(len(T), 1500)])
                    if campaign and not boycott:
                        pools["contested or slack cells, equal weights"].append(blk[rng.choice(len(T), 1500)])
                for wlab, pool in pools.items():
                    if pool:
                        summ.append(dict(rho=rho, t_min=tm, prior=pr.label, weighting=wlab, **_pooled_stats(np.vstack(pool))))
                keep[(rho, pr.label, tm)] = pools
    # two-floor mixture: boycott cells at a low floor (the failed votes) and contested or slack cells at t_min = 0.35
    # (the votes that met the quorum), weighted 38 : 39 as in the data
    for (rho, plab, tm), pools in list(keep.items()):
        if tm not in (0.05, 0.10):
            continue
        hi = keep.get((rho, plab, 0.35), {}).get("contested or slack cells, equal weights", [])
        lo = pools.get("boycott cells, equal weights", [])
        if hi and lo:
            A, B = np.vstack(lo), np.vstack(hi)
            nA, nB = 38 * 400, 39 * 400
            M = np.vstack([A[rng.choice(len(A), nA)], B[rng.choice(len(B), nB)]])
            summ.append(dict(rho=rho, t_min=f"{tm:g} (boycott) / 0.35 (contested)", prior=plab,
                             weighting="two-floor mixture, 38 : 39", **_pooled_stats(M)))
    report.write_csv(cells, "vii_R4_italy_cells", P, scen,
                     note="R4: Italian analogue, V2 at 1/2 plus majority (T1). Non-resident turnout 0.78 of resident turnout, "
                          "mu_A = 1/2; Monte Carlo 6,000 draws per cell (mu ~ prior, A11 noise; draws with no ballots dropped); "
                          "weight = share of the 77 Italian questions whose observed turnout rounds to the cell's tau.")
    report.write_csv(summ, "vii_R4_italy_pooled", P, scen,
                     note="R4: statistics pooled over the tau grid, computed as in Table VII.9 (void = realised turnout below 1/2 "
                          "of E), under three weightings: the Italian observed-turnout distribution (as requested; biased toward "
                          "locked cells, because a boycott pushes observed turnout below the baseline), equal weights on the cells "
                          "where the challenger campaigns, and equal weights on the boycott cells. Data: share void 0.494; Yes "
                          "majority in 37/38 void; median Yes in void 0.793; r void +0.33; r valid -0.31; r pooled -0.41.")
    return cells, summ


# ---------------------------------------------------------------- R5 2026 timing
DEPOSITS = {"30.10.2026": (2026, 303), "30.11.2026": (2026, 334), "15.01.2027": (2027, 15), "31.03.2027": (2027, 90)}


def run_R5(P, scen):
    Pm = pipeline._mod()
    rng = np.random.default_rng(0)
    rows = []
    FR0 = pipeline.kqz_false_rejection("R0")
    for opt in ("O0", "O9"):
        cal = Pm.Calendar(Pm.gen_timeline(0.0, rng), opt, -Pm.WY * 60, Pm.WY * 80)
        for pres in ("earliest", "latest"):
            p = dict(Pm.BASE)
            p.update(FR=FR0, president=pres)
            gap = p["DEC"] + p["GMIN"]
            for lab, (yr, doy) in DEPOSITS.items():
                dep = (yr - 2025) * Pm.WY + (doy - 1) // 7
                if dep % Pm.WY > Pm.W_DEP_LAST:
                    dep = (dep // Pm.WY + 1) * Pm.WY
                dist = {}
                for cure, pc in ((1, p["p_cure"]), (0, 1 - p["p_cure"])):
                    c = dep + p["KQZ"] + (p["CURE"] if cure else 0) + p["CC"]
                    tp = Pm.schedule(cal, c, opt, gap, pres)
                    dist[tp] = dist.get(tp, 0.0) + pc
                ks = np.array(sorted(dist)); ps = np.array([dist[k] for k in ks]); ps = ps / ps.sum()
                yrs = 2025 + ks // Pm.WY
                rows.append(dict(option=opt, timing=pipeline.TS_LABELS[opt], president=pres, deposit=lab,
                                 deposit_week_abs=dep,
                                 Pr_poll_2027=float(ps[yrs == 2027].sum()), Pr_poll_2028=float(ps[yrs == 2028].sum()),
                                 Pr_poll_later=float(ps[yrs > 2028].sum()),
                                 Pr_vote_12m=float(ps[ks - dep <= 52].sum()), Pr_vote_18m=float(ps[ks - dep <= 78].sum()),
                                 Pr_vote_24m=float(ps[ks - dep <= 104].sum()),
                                 mean_months=float(((ks - dep) * ps).sum() * 12 / Pm.WY),
                                 polling_weeks=";".join(f"{2025 + k // Pm.WY}-w{k % Pm.WY}" for k in ks),
                                 Pr_valid_list_passes_KQZ=1 - FR0))
    report.write_csv(rows, "vii_R5_2026_timing", P, scen,
                     note="R5: valid national initiative deposited on the stated date; statutory maximum stage times of Section V "
                          "Part 1 (KQZ 13 weeks, cure 7 weeks with probability 1/2, Court 9 weeks, decree 5, lead 6); regular "
                          "terms (no early election: lambda_ee = 0, a tool gap); cut-off tested on the Court's ruling; R0 "
                          "false rejection at the baseline list applied as a separate factor (column Pr_valid_list_passes_KQZ). "
                          "Week = 7-day block of the idealised 52-week year.")
    return rows


# ---------------------------------------------------------------- R6 verification of the 2026 lists
def admit_R0(J, VL, S=50000):
    n = int(math.ceil(0.05 * J))
    thr = math.ceil(n * S / J - 1e-9)
    return float(stats.hypergeom.sf(thr - 1, J, VL, n)), n


def admit_R3(J, VL, S=50000, nfix=1500):
    if J <= 3000:
        return (1.0 if VL >= S else 0.0), J
    n = nfix
    x = np.arange(0, n + 1)
    pmf = stats.hypergeom.pmf(x, J, VL, n)
    a = np.where(x / n >= 0.92, 0.02, 0.05)
    proj = J * x / n
    acc = proj >= (1 + a) * S
    rej = proj < (1 - a) * S
    band = ~(acc | rej)
    pb = float(pmf[band].sum())
    return float(pmf[acc].sum() + pb * (1.0 if VL >= S else 0.0)), n + pb * J


def run_R6(P, scen):
    rows = []
    S = 50000
    for J in (35000, 50000, 52000, 55000, 60000, 70000):
        for v in (0.80, 0.85, 0.90, 0.95):
            VL = int(round(v * J))
            for rule, f in (("R0", admit_R0), ("R3", admit_R3)):
                pa, lines = f(J, VL, S)
                rows.append(dict(J=J, v_L=v, valid_lines=VL, margin=VL / S - 1, rule=rule, Pr_admit=pa,
                                 FR=(1 - pa) if VL >= S else 0.0, FA=pa if VL < S else 0.0, lines_examined=lines))
    report.write_csv(rows, "vii_R6_2026_verification", P, scen,
                     note="R6: S = 50,000; R0 = 5% random sample, admit iff the projection reaches S (no band); R3 = F44 band "
                          "rule (1,500 lines; admit at 102% or 105% of S, reject below 98% or 95%, count every line between). "
                          "FR = Pr[reject | valid lines >= S]; FA = Pr[admit | valid lines < S]; lines = expected lines examined.")
    return rows


# ---------------------------------------------------------------- R7 2026 outcome under Regime A, and R8 draft rules
def _grid_rows(P, rules, case, Rs=(1.89e6, 1.92e6, 1.95e6), taus=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
               tmins=(0.0, 0.20, 0.35), c4=True):
    rows = []
    pv0 = pipeline.pr_vote("O0", "earliest", FR=pipeline.kqz_false_rejection("R0"))["Pr_vote_24"]
    pv5 = pipeline.pr_vote("O9", "earliest", FR=pipeline.kqz_false_rejection("R3"))["Pr_vote_24"]
    priors = [("A2", 0.0, 1.0), ("A2", 0.3, 1.0), ("Beta", 0.0, 0.5), ("Beta", 0.0, 2.5)]
    for lab, rule in rules:
        for den in ("E_full", "E_dom"):
            for R in Rs:
                el = national(P.override(**{"electorate.R": R}), den)
                for tau in taus:
                    for tm in tmins:
                        for fam, pi, a in priors:
                            g = Game(rule, el, tau=tau, t_min=tm, prior=prior_of(fam, pi, 0.5, a), pi=pi)
                            base = dict(case=case, package_rule=lab, rule=rule.describe(), denominator=den, R=R,
                                        rho=el.rho, tau=tau, t_min=tm, pi=pi, prior=g.prior.label)
                            if not g.feasible:
                                rows.append(dict(base, feasible=False)); continue
                            _, r = solve_row(g)
                            row = dict(base, feasible=True, locked=r["regime"].startswith("lock"), **r)
                            if c4:
                                row.update(C4_opposed_P0=pv0 * max(0.0, r["P_dec"] - r["P_change"]),
                                           C4_opposed_P5=pv5 * r["P_dec"], Pr_vote_24_TS0=pv0, Pr_vote_24_TS9=pv5)
                            rows.append(row)
    return rows


def run_R7(P, scen):
    rows = _grid_rows(P, [("P0 T1, V3 at 1/3 of E", V3(q=1 / 3))], "2026 Regime A")
    report.write_csv(rows, "vii_R7_2026_regimeA", P, scen,
                     note="R7: T1 under Regime A (P0) at 2026 values; E_full 3,713,761 or E_dom 3,467,826; R 1.89/1.92/1.95M. "
                          "C4 against an opposed majority: P0 = Pr_vote(24 | TS0, R0) x max(0, Pr_dec - Pr_change); P5 = "
                          "Pr_vote(24 | TS9, R3) x Pr_dec (substance mooting rule). Beta priors at pi = 0 only.")
    return rows


DRAFT_RULES = {
    "P3b T1 (Yes reading): V2 at 0.40 + Yes >= 0.30 E + majority": V8x(q_P=0.40, q_A=0.30),
    "P3c T1 (valid-vote reading): V2 at 0.40 + valid >= 0.30 E + majority (= V2 at 0.40)": V8x(q_P=0.40),
    "Draft T4: V2 at 0.50 + majority": V2(q=0.50),
    "Draft T3 (Yes reading): V11 carrying V2 at 0.30 + Yes >= 0.25 E": V11(base=V8x(q_P=0.30, q_A=0.25)),
    "Draft T3 (valid-vote reading): V11 carrying V2 at 0.30": V11(base=V8x(q_P=0.30)),
}


def run_R8(P, scen):
    from .evaluate import package_uncertainty, evaluate_package
    el = national(P, "E_full")
    lb = []
    for lab, r in DRAFT_RULES.items():
        for tm in (0.0, 0.20, 0.35):
            t = tau_lock(r, el, tm)
            lb.append(dict(rule=lab, t_min=tm, lock_boundary_tau=t if t is not None else "locked at every tau up to the feasible maximum"))
    report.write_csv(lb, "vii_R8_draft_lock_boundaries", P, scen,
                     note="R8: lowest stand-alone turnout (share of E_full) at which the challenger campaigns; pi = 0, A2.")
    rows = _grid_rows(P, [(k, v) for k, v in DRAFT_RULES.items() if "T1" in k], "March 2026 draft, T1", c4=False)
    report.write_csv(rows, "vii_R8_draft_grid", P, scen, note="R8: the draft's T1 readings on the R7 grid.")
    cen, unc, drw = [], [], []
    for pk in ("P3b", "P3c"):
        cen += evaluate_package(P, pk)
        s, d = package_uncertainty(P, pk, n_draws=60, seed=7, types=("T1", "T3", "T4", "T5"), return_draws=True)
        unc += s
        drw += d
    report.write_csv(cen, "vii_R8_packages_central", P, scen, note="R8: packages P3b and P3c at the central cell (as Table VI.6).")
    report.write_csv(unc, "vii_R8_packages_uncertainty", P, scen, note="R8: the 60 common draws of F70 (seed 7).")
    report.write_csv(drw, "vii_R8_packages_draws", P, scen, note="R8: draws.")
    return lb, rows, cen, unc, drw


# ---------------------------------------------------------------- R9 comparators
COMPARATORS = [
    ("Denmark 1939", "T4", V3(q=0.45), 0.489, (1.0,)),
    ("Lithuania 2024 (co-scheduled)", "T4", V3(q=0.50), 0.595, (1.0, 1.1, 1.2)),
    ("Colombia 2016", "T3", V3(q=0.13), 0.374, (1.0,)),
    ("Colombia 2018", "T3", V2(q=1 / 3), 0.321, (1.0,)),
    ("Montenegro 2006", "T3", V8(q=0.50, qS=0.55, limb="participation"), 0.865, (1.0,)),
    ("Hungary 2016", "T3", V2(q=0.50), 0.44, (1.0,)),
    ("Netherlands 2016", "T5", V11(base=V2(q=0.30)), 0.323, (1.0,)),
]


def run_R9(P, scen):
    rows = []
    for name, typ, rule, tau, rhos in COMPARATORS:
        for rho in rhos:
            el = Electorate(E=rho, R=1.0, label=name)
            for tm in (0.10, 0.20):
                g = Game(rule, el, tau=tau, t_min=tm, prior=uniform(0.0), pi=0.0)
                base = dict(case=name, type=typ, rule=rule.describe(), rho=rho, tau=tau, t_min=tm)
                if not g.feasible:
                    rows.append(dict(base, feasible=False)); continue
                _, r = solve_row(g)
                rows.append(dict(base, feasible=True, **r))
    report.write_csv(rows, "vii_R9_comparators", P, scen,
                     note="R9: comparator regimes, illustrative (A2, pi = 0; rho = 1 unless stated; observed turnout as the "
                          "stand-alone baseline). Hungary's valid-vote rule is run as V2 (no invalid ballots in the model).")
    return rows


# ---------------------------------------------------------------- R10 V10 override frequency
N_REF_UNITS = 61  # the 2014 map; R10's draws are aligned to it (v1.1.5)


def run_R10(P, scen, n=400_000, seed=11):
    m = load_municipalities()
    q = m.qarks(R_now=1.92e6)
    wq = np.array([q[k]["E"] for k in sorted(q)])
    wm = np.array(m.E_hat, float)
    el = national(P, "E_full")
    t0 = el.t0(P.behaviour.tau_central)
    rows = []
    rng = np.random.default_rng(seed)
    for fam, a in (("A2", 1.0), ("Beta", 0.5), ("Beta", 2.5)):
        pr = prior_of(fam, 0.0, 0.5, a)
        nm = len(m.names)
        for lab, w, k in (("P4: 7 of 12 qarks (E-hat weights)", wq, 7),
                          (f"Swiss-type: majority of the {nm} municipalities", wm, nm // 2 + 1)):
            for su in (0.03, 0.05, 0.08):
                mu = pr.sample(rng, n)
                sig = el.rho * SIG
                Z1 = rng.standard_normal(n)
                Z2 = THETA * Z1 + math.sqrt(1 - THETA ** 2) * rng.standard_normal(n)
                X = np.clip(t0 + sig * Z1, 0, 1)
                D = np.clip(t0 + sig * Z2, 0, 1)
                ww = w / w.sum()
                over = 0; nat_yes = 0
                for c0 in range(0, n, 50_000):
                    sl = slice(c0, c0 + 50_000)
                    mu_u = np.clip(mu[sl, None] + su * rng.standard_normal((len(mu[sl]), len(w))), 0, 1)
                    if w is wm and len(w) < N_REF_UNITS:
                        # v1.1.5: keep the shared stream aligned with the 61-unit map, so that the later rows
                        # (P4 under the next priors) use the same draws on every municipal map
                        rng.standard_normal((len(mu[sl]), N_REF_UNITS - len(w)))
                    Yu = mu_u * X[sl, None] * ww
                    Nu = (1 - mu_u) * D[sl, None] * ww
                    nat = Yu.sum(1) > Nu.sum(1)
                    wins = (Yu > Nu).sum(1)
                    over += int(np.sum(nat & (wins < k)))
                    nat_yes += int(np.sum(nat))
                rows.append(dict(rule=lab, prior=pr.label, sigma_u=su, n=n, Pr_override=over / n,
                                 Pr_override_given_national_Yes=over / max(nat_yes, 1), Pr_national_Yes=nat_yes / n))
    report.write_csv(rows, "vii_R10_V10_override", P, scen,
                     note="R10: share of votes in which a national Yes majority fails the territorial limb (central cell: "
                          "tau = 0.30 of E_full, t_min = 0.20, V1-equivalent strategies s = t0 = 0.580 for both sides; A11 noise; "
                          "unit support mu_u = mu + sigma_u eps_u, independent; units weighted by E-hat). Swiss comparison: "
                          "10 of 637 proposals (1.6%) failed only on the cantons.")
    return rows


def run_all_cases(P, scen):
    out = {}
    for f in (run_R1, run_R2, run_R3, run_R4, run_R5, run_R6, run_R7, run_R8, run_R9, run_R10):
        out[f.__name__] = f(P, scen)
    return out

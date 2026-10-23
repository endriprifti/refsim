"""Electorate module (Specification module 1).

Denominators, effective resident bars and municipal proxies.
All bars are in resident units (share of R), as in Section III:
    rho = E/R;  t0 = rho*tau - tau_ab*(rho - 1)                     (D2)
    Q_A = q*rho - mu_A*tau_ab*(rho - 1)       approval against E    (3.7)
    Q_P = q*rho - tau_ab*(rho - 1)            participation against E
    Q_R = q*rho - mu_A*tau_ab*(rho - 1)       rejection against E (V4)
    V5:  Q = q_T * T_ref / R  (approval or participation against the reference election)
    V6d: Q = q (approval against R itself); V6a/b/c: q * E_variant / R
Municipal proxies (eq. II.2): E_hat_u = lambda_E * CR_u, R_hat_u = lambda_R * C_u;
D11 rebases R_hat_u from the 2023 census to the simulation date.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .params import DATA_DIR


@dataclass
class Electorate:
    E: float                       # registered electorate used as denominator
    R: float                       # resident electorate able to vote
    tau_ab: float = 0.0            # turnout among registered non-residents (A1)
    mu_A: float = 0.5              # non-resident share for the non-default outcome
    label: str = ""
    tauR_ref: float = 0.76         # T_ref/R at the reference election (V5; D2)

    @property
    def rho(self) -> float:
        return self.E / self.R

    @property
    def n_ab(self) -> float:
        """Non-resident votes in resident units: tau_ab*(rho - 1)."""
        return self.tau_ab * (self.rho - 1.0)

    def t0(self, tau: float) -> float:
        """Baseline resident turnout from registered turnout tau (share of E)."""
        return self.rho * tau - self.n_ab

    def tau(self, t0: float) -> float:
        return (t0 + self.n_ab) / self.rho

    # ---- effective bars (resident units)
    def Q_approval(self, q: float) -> float:
        return q * self.rho - self.mu_A * self.n_ab

    def Q_participation(self, q: float) -> float:
        return q * self.rho - self.n_ab

    def Q_resident(self, q: float) -> float:
        """V6d: approval against R itself (non-resident votes still count for the side)."""
        return q - self.mu_A * self.n_ab

    def b_ab(self) -> float:
        """Non-resident skew of the majority condition, (1 - 2 mu_A) tau_ab (rho - 1)."""
        return (1.0 - 2.0 * self.mu_A) * self.n_ab


def national(params, denominator: str = "E_full", R: float | None = None, tau_ab: float | None = None) -> Electorate:
    e = params.electorate
    Rv = e.R if R is None else R
    if denominator == "E_full":
        E = e.E_full
    elif denominator == "E_dom":
        E = e.E_full - e.E_ocv
    elif denominator == "R":
        E = Rv
    else:
        raise ValueError(denominator)
    return Electorate(E=float(E), R=float(Rv), tau_ab=e.tau_ab if tau_ab is None else tau_ab,
                      mu_A=e.mu_A, label=denominator, tauR_ref=e.tauR_ref)


# ---------------------------------------------------------------- drift path (A29, D11)
def rho_path(t_years, E0: float, R0: float, gamma_E: float, d_R: float):
    """Deterministic A29 path rho(t) = (E0 + gamma_E t) / (R0 exp(-d_R t))."""
    t = np.asarray(t_years, float)
    return (E0 + gamma_E * t) / (R0 * np.exp(-d_R * t))


def gamma_rho(E0: float, gamma_E: float, d_R: float, t: float = 0.0) -> float:
    return gamma_E / (E0 + gamma_E * t) + d_R


def t_cross(level: float, E0: float, R0: float, gamma_E: float, d_R: float) -> float:
    """Years until the deterministic path reaches `level` (0 if already above)."""
    from scipy import optimize
    if rho_path(0.0, E0, R0, gamma_E, d_R) >= level:
        return 0.0
    return float(optimize.brentq(lambda x: rho_path(x, E0, R0, gamma_E, d_R) - level, 0.0, 300.0))


# ---------------------------------------------------------------- municipal proxies (Table C.4)
@dataclass
class Municipalities:
    names: list
    qark: list
    C: np.ndarray          # census population 2023
    CR: np.ndarray         # civil-registry population
    E_hat: np.ndarray      # 2025 list proxy
    R_hat_2023: np.ndarray # 2023 census-basis resident adults
    min_sig: np.ndarray
    R_2023: float = 1.93e6
    meta: dict = field(default_factory=dict)

    def R_hat(self, R_now: float) -> np.ndarray:
        """D11: rebase R_hat from the 2023 census basis to R(t) (lambda_R(t) = R(t)/P_cen)."""
        return self.R_hat_2023 * (R_now / self.R_2023)

    def rho_u(self, R_now: float | None = None) -> np.ndarray:
        if R_now is None:
            return self.E_hat / self.R_hat_2023
        return self.E_hat / self.R_hat(R_now)

    def count_above(self, q: float = 1 / 3, R_now: float | None = None, years: float = 0.0,
                    gamma: float = 0.0) -> int:
        """Units where the approval bar q*rho_u exceeds 1 (change infeasible even at full turnout)."""
        r = self.rho_u(R_now) * math.exp(gamma * years)
        return int(np.sum(q * r > 1.0))

    def remark1_exceptions(self, R_now: float | None = None, years: float = 0.0, gamma: float = 0.0):
        """Units with rho_u < 1.5 (Q < t_bar/2 at q = 1/3): Remark 1 does not apply."""
        r = self.rho_u(R_now) * math.exp(gamma * years)
        return [(self.names[i], float(r[i])) for i in np.argsort(r) if r[i] < 1.5]

    def qarks(self, R_now: float | None = None):
        out = {}
        Rh = self.R_hat_2023 if R_now is None else self.R_hat(R_now)
        for q in sorted(set(self.qark)):
            idx = [i for i, k in enumerate(self.qark) if k == q]
            out[q] = dict(E=float(self.E_hat[idx].sum()), R=float(Rh[idx].sum()),
                          rho=float(self.E_hat[idx].sum() / Rh[idx].sum()),
                          largest_share=float(self.C[idx].max() / self.C[idx].sum()))
        return out


# v1.1.5: two municipal maps. "2014": the 61 municipalities of Law No. 115/2014 (Table C.4 as printed); "2027": the 46
# municipalities of Law No. 100/2026 (adopted 9 October 2026; the 2027 local elections are held on it), aggregated from
# Table C.4 by whole-municipality mergers within the same qark (column "members"). The qarqe are unchanged. Default "2027";
# set REFSIM_MAP=2014 or call set_municipal_map("2014") (run.py --map 2014) to reproduce the 61-unit results of v1.1.4.
MUNICIPAL_MAPS = {
    "2014": ("table_c4_municipalities.csv", "Law No. 115/2014: 61 municipalities (Section II, Table C.4)"),
    "2027": ("table_c4_municipalities_2027.csv",
             "Law No. 100/2026: 46 municipalities from the 2027 local elections (Table C.4 aggregated)"),
}
_MAP = {"key": os.environ.get("REFSIM_MAP", "2027")}


def set_municipal_map(key: str) -> None:
    if key not in MUNICIPAL_MAPS:
        raise ValueError(f"unknown municipal map {key!r}; use one of {sorted(MUNICIPAL_MAPS)}")
    _MAP["key"] = key
    try:                                   # qark weights are cached; the qarqe are the same on both maps, but clear anyway
        from .evaluate import qark_weights
        qark_weights.cache_clear()
    except Exception:
        pass


def municipal_map() -> str:
    return _MAP["key"]


def load_municipalities(path: str | None = None, R_2023: float = 1.93e6, map: str | None = None) -> Municipalities:
    key = map or _MAP["key"]
    fname, label = MUNICIPAL_MAPS[key]
    path = path or os.path.join(DATA_DIR, fname)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return Municipalities(names=[r["municipality"] for r in rows], qark=[r["qark"] for r in rows],
                          C=f("census_pop_2023"), CR=f("civil_registry_pop"), E_hat=f("E_hat_2025"),
                          R_hat_2023=f("R_hat_2023basis"), min_sig=f("min_signatures").astype(int),
                          R_2023=R_2023, meta=dict(source=label, map=key, path=path))


def local_min_signatures(E_u: float) -> int:
    """Law 9087 Art. 132(1): 10% of local voters or 20,000, whichever is smaller."""
    return 20000 if E_u >= 200000 else int(math.ceil(0.1 * E_u))


def adult_decline_from_instat(path: str | None = None, deduction_15_17: float = 100000.0) -> dict:
    """D11: decline of the 15+ and (approximately) 18+ population from INSTAT dependency ratios."""
    path = path or os.path.join(DATA_DIR, "instat_population.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    p15 = []
    for r in rows:
        yd, od = float(r["youth_dependency_pct"]), float(r["old_age_dependency_pct"])
        share_0_14 = (yd / 100) / (1 + (yd + od) / 100)
        p15.append(float(r["resident_population"]) * (1 - share_0_14))
    tot = [float(r["resident_population"]) for r in rows]
    return dict(d_total=1 - tot[1] / tot[0], d_15plus=1 - p15[1] / p15[0],
                d_18plus=1 - (p15[1] - deduction_15_17) / (p15[0] - deduction_15_17),
                pop15=p15)

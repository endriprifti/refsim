"""Validity rules V1-V11 as plug-ins with a common interface (Specification module 3; menu of Section I).

Every rule offers two views of the same legal test:

1. `intervals(sx, sd, el)`: the set of resident support mu at which the non-default option
   wins, given side turnout rates sx (challenger) and sd (default) in resident units
   (Section III's group model; vectorised in sx, sd).  Returns a list of Piece(a, b, weight, kind):
   kind 'change' (binding) or 'advisory'; `weight` is the payoff value of that outcome to the
   challenger relative to a binding change (1 for binding, lambda_adv for advisory).
2. `evaluate(v)`: the ex-post outcome from vote counts v = Votes(Y, N, E, R, T_ref, units),
   returning (outcome, valid) with outcome in {'change', 'advisory', 'default'} and valid=True when
   the vote produced a valid result for one side (C4, eq. (I.10)).

Orientation: Y counts votes for the NON-DEFAULT outcome (repeal/approval under Regime A;
rejection of the amendment under Regime B, T5), N votes for the default.

To add a rule: subclass Rule, implement `intervals`, `default_valid` and `evaluate`, set
`default_monotone`, and register it in RULES (see docs/MANUAL.md, section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

INF = 9.0


@dataclass
class Piece:
    a: np.ndarray
    b: np.ndarray
    weight: float = 1.0
    kind: str = "change"


@dataclass
class Votes:
    Y: np.ndarray            # votes for the non-default outcome
    N: np.ndarray            # votes for the default
    E: float                 # registered electorate (the rule's E-denominator)
    R: float                 # residents able to vote
    T_ref: float = np.nan    # ballots at the reference election (V5)
    units_Y: Optional[np.ndarray] = None   # (n, m_U) unit-level counts for V10
    units_N: Optional[np.ndarray] = None
    E_alt: Optional[dict] = None           # other denominators by name (E_full, E_dom, E_c)


# ---------------------------------------------------------------- interval primitives (vectorised)
def _arr(x):
    return np.asarray(x, float)


def iv_majority(sx, sd, b_ab=0.0):
    """Y > N  <=>  mu > (sd + b_ab)/(sx + sd)."""
    sx, sd = _arr(sx), _arr(sd)
    den = sx + sd
    a = np.where(den > 0, (sd + b_ab) / np.where(den > 0, den, 1.0), INF)
    return a, np.full_like(a, INF)


def iv_approval(Q, sx):
    """mu*sx >= Q."""
    sx = _arr(sx)
    a = np.where(sx > 0, Q / np.where(sx > 0, sx, 1.0), INF)
    return a, np.full_like(a, INF)


def iv_participation(Q, sx, sd):
    """sd + (sx - sd) mu >= Q (turnout bar, resident units, non-residents netted in Q)."""
    sx, sd = np.broadcast_arrays(_arr(sx), _arr(sd))
    d = sx - sd
    a = np.full(d.shape, INF)
    b = np.full(d.shape, INF)
    up = d > 1e-12
    dn = d < -1e-12
    eq = ~(up | dn)
    a[up] = (Q - sd[up]) / d[up]
    a[dn] = -INF
    b[dn] = (sd[dn] - Q) / (sd[dn] - sx[dn])
    ok = eq & (sd >= Q)
    a[ok] = -INF
    return a, b


def iv_supermajority(q, sx, sd, n_ab=0.0, mu_A=0.5):
    """Y >= q (Y + N)  <=>  mu >= (q sd + (q - mu_A) n_ab) / ((1 - q) sx + q sd)."""
    sx, sd = _arr(sx), _arr(sd)
    den = (1 - q) * sx + q * sd
    a = np.where((den > 0) & (sx > 0), (q * sd + (q - mu_A) * n_ab) / np.where(den > 0, den, 1.0), INF)
    return a, np.full_like(a, INF)


def inter(*ivs):
    """Intersection of intervals (vectorised); empty sets are returned as (INF, INF)."""
    a, b = _arr(ivs[0][0]), _arr(ivs[0][1])
    for ai, bi in ivs[1:]:
        a = np.maximum(a, ai)
        b = np.minimum(b, bi)
    empty = a >= b
    return np.where(empty, INF, a), np.where(empty, INF, b)


def below(x):
    x = _arr(x)
    return np.full_like(x, -INF), x


# ---------------------------------------------------------------- base class
class Rule:
    code = "V?"
    name = ""
    default_monotone = True
    symmetric_validity = False    # True: the default also needs to meet the bar to be a valid result
    binding = True

    def __init__(self, **params):
        self.params = params

    def describe(self) -> str:
        args = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.code} {self.name}" + (f" ({args})" if args else "")

    # equilibrium view
    def intervals(self, sx, sd, el) -> list:
        raise NotImplementedError

    def default_valid(self, sx, sd, el):
        """Interval of mu where the default wins with a valid result; default: whenever change fails."""
        a = np.min([p.a for p in self.intervals(sx, sd, el)], axis=0)
        return below(a)

    # ex-post view
    def evaluate(self, v: Votes):
        raise NotImplementedError


def _den(v: Votes, which: str):
    if which in (None, "E"):
        return v.E
    if which == "R":
        return v.R
    return v.E_alt[which]


# ---------------------------------------------------------------- V1 no quorum
class V1(Rule):
    code, name = "V1", "no quorum (majority of valid votes)"

    def intervals(self, sx, sd, el):
        a, b = iv_majority(sx, sd, el.b_ab())
        return [Piece(a, b)]

    def evaluate(self, v):
        ch = v.Y > v.N
        return np.where(ch, "change", "default"), v.Y != v.N


# ---------------------------------------------------------------- V2 participation quorum
class V2(Rule):
    code, name = "V2", "participation quorum"
    default_monotone = False
    symmetric_validity = True

    def __init__(self, q=0.40):
        super().__init__(q=q)

    def intervals(self, sx, sd, el):
        Q = el.Q_participation(self.params["q"])
        return [Piece(*inter(iv_majority(sx, sd, el.b_ab()), iv_participation(Q, sx, sd)))]

    def default_valid(self, sx, sd, el):
        Q = el.Q_participation(self.params["q"])
        mv, _ = iv_majority(sx, sd, el.b_ab())
        return inter(below(mv), iv_participation(Q, sx, sd))

    def evaluate(self, v):
        T = v.Y + v.N
        ok = T >= self.params["q"] * v.E
        return np.where(ok & (v.Y > v.N), "change", "default"), ok & (v.Y != v.N)


# ---------------------------------------------------------------- V3 approval quorum (Regime A)
class V3(Rule):
    code, name = "V3", "approval quorum"
    symmetric_validity = True

    def __init__(self, q=1 / 3, denominator="E"):
        super().__init__(q=q, denominator=denominator)

    def _Q(self, el):
        return el.Q_approval(self.params["q"])

    def intervals(self, sx, sd, el):
        return [Piece(*inter(iv_majority(sx, sd, el.b_ab()), iv_approval(self._Q(el), sx)))]

    def default_valid(self, sx, sd, el):
        """N > Y and N >= bar: (1 - mu) sd >= Q_d, Q_d = q rho - (1 - mu_A) n_ab."""
        sd = _arr(sd)
        Qd = self.params["q"] * el.rho - (1 - el.mu_A) * el.n_ab
        lim = np.where(sd > 0, 1.0 - Qd / np.where(sd > 0, sd, 1.0), -INF)
        mv, _ = iv_majority(sx, sd, el.b_ab())
        return below(np.minimum(mv, lim))

    def evaluate(self, v):
        bar = self.params["q"] * _den(v, self.params.get("denominator"))
        ch = (v.Y > v.N) & (v.Y >= bar)
        dv = (v.N > v.Y) & (v.N >= bar)
        return np.where(ch, "change", "default"), ch | dv


# ---------------------------------------------------------------- V4 rejective quorum (Regime B, T5)
class V4(Rule):
    code, name = "V4", "rejective quorum (Regime B: amendment falls iff more than q of E vote against)"

    def __init__(self, q=0.5):
        super().__init__(q=q)

    def intervals(self, sx, sd, el):
        Q = el.Q_approval(self.params["q"])   # Q_R = q_R rho - mu_A tau_ab (rho - 1)
        a, b = iv_approval(Q, sx)
        if self.params["q"] < 0.5:            # F18: also N > Y
            return [Piece(*inter((a, b), iv_majority(sx, sd, el.b_ab())))]
        return [Piece(a, b)]

    def evaluate(self, v):
        ch = v.Y > self.params["q"] * v.E
        if self.params["q"] < 0.5:
            ch = ch & (v.Y > v.N)
        return np.where(ch, "change", "default"), np.ones_like(ch, bool)


# ---------------------------------------------------------------- V5 quorum tied to last-election turnout
class V5(Rule):
    code, name = "V5", "quorum tied to the last election's turnout"

    def __init__(self, qT=0.5, form="approval"):
        super().__init__(qT=qT, form=form)
        self.default_monotone = form == "approval"
        self.symmetric_validity = True
        self.code = "V5a" if form == "approval" else "V5p"

    def _Q(self, el):
        # resident units: q_T * T_ref / R = q_T * tauR_ref
        return self.params["qT"] * el.tauR_ref

    def intervals(self, sx, sd, el):
        Q = self._Q(el)
        if self.params["form"] == "approval":
            return [Piece(*inter(iv_majority(sx, sd, el.b_ab()), iv_approval(Q - el.mu_A * el.n_ab, sx)))]
        return [Piece(*inter(iv_majority(sx, sd, el.b_ab()), iv_participation(Q - el.n_ab, sx, sd)))]

    def default_valid(self, sx, sd, el):
        Q = self._Q(el)
        mv, _ = iv_majority(sx, sd, el.b_ab())
        if self.params["form"] == "approval":
            sd = _arr(sd)
            lim = np.where(sd > 0, 1.0 - (Q - (1 - el.mu_A) * el.n_ab) / np.where(sd > 0, sd, 1.0), -INF)
            return below(np.minimum(mv, lim))
        return inter(below(mv), iv_participation(Q - el.n_ab, sx, sd))

    def evaluate(self, v):
        bar = self.params["qT"] * v.T_ref
        if self.params["form"] == "approval":
            ch = (v.Y > v.N) & (v.Y >= bar)
            return np.where(ch, "change", "default"), ch | ((v.N > v.Y) & (v.N >= bar))
        ok = (v.Y + v.N) >= bar
        return np.where(ok & (v.Y > v.N), "change", "default"), ok & (v.Y != v.N)


# ---------------------------------------------------------------- V6 resident-indexed quorum (variants V6a-V6d, D8)
class V6(V3):
    code, name = "V6", "resident-indexed approval quorum"

    def __init__(self, q=1 / 3, variant="d"):
        Rule.__init__(self, q=q, variant=variant)
        self.code = "V6" + variant

    def _Q(self, el):
        if self.params["variant"] == "d":
            return el.Q_resident(self.params["q"])
        return el.Q_approval(self.params["q"])   # V6a/b/c: pass an Electorate built on that denominator

    def intervals(self, sx, sd, el):
        return [Piece(*inter(iv_majority(sx, sd, el.b_ab()), iv_approval(self._Q(el), sx)))]

    def default_valid(self, sx, sd, el):
        sd = _arr(sd)
        if self.params["variant"] == "d":
            Qd = self.params["q"] - (1 - el.mu_A) * el.n_ab
        else:
            Qd = self.params["q"] * el.rho - (1 - el.mu_A) * el.n_ab
        lim = np.where(sd > 0, 1.0 - Qd / np.where(sd > 0, sd, 1.0), -INF)
        mv, _ = iv_majority(sx, sd, el.b_ab())
        return below(np.minimum(mv, lim))

    def evaluate(self, v):
        den = v.R if self.params["variant"] == "d" else v.E
        bar = self.params["q"] * den
        ch = (v.Y > v.N) & (v.Y >= bar)
        return np.where(ch, "change", "default"), ch | ((v.N > v.Y) & (v.N >= bar))


# ---------------------------------------------------------------- V7 supermajority of votes cast
class V7(Rule):
    code, name = "V7", "supermajority of valid votes"

    def __init__(self, qS=0.6):
        super().__init__(qS=qS)

    def intervals(self, sx, sd, el):
        return [Piece(*iv_supermajority(self.params["qS"], sx, sd, el.n_ab, el.mu_A))]

    def evaluate(self, v):
        T = v.Y + v.N
        ch = (T > 0) & (v.Y >= self.params["qS"] * T)
        return np.where(ch, "change", "default"), T > 0


# ---------------------------------------------------------------- V8 combined rule
class V8(Rule):
    code, name = "V8", "combined rule (approval limb + supermajority, or participation limb)"
    symmetric_validity = True

    def __init__(self, q=1 / 3, qS=0.55, limb="approval"):
        super().__init__(q=q, qS=qS, limb=limb)
        self.default_monotone = limb == "approval"

    def intervals(self, sx, sd, el):
        sup = iv_supermajority(self.params["qS"], sx, sd, el.n_ab, el.mu_A)
        if self.params["limb"] == "approval":
            return [Piece(*inter(sup, iv_approval(el.Q_approval(self.params["q"]), sx)))]
        return [Piece(*inter(sup, iv_participation(el.Q_participation(self.params["q"]), sx, sd)))]

    def default_valid(self, sx, sd, el):
        a = self.intervals(sx, sd, el)[0].a
        if self.params["limb"] == "approval":
            sd = _arr(sd)
            Qd = self.params["q"] * el.rho - (1 - el.mu_A) * el.n_ab
            lim = np.where(sd > 0, 1.0 - Qd / np.where(sd > 0, sd, 1.0), -INF)
            return below(np.minimum(a, lim))
        return inter(below(a), iv_participation(el.Q_participation(self.params["q"]), sx, sd))

    def evaluate(self, v):
        T = v.Y + v.N
        sup = (T > 0) & (v.Y >= self.params["qS"] * T)
        if self.params["limb"] == "approval":
            ch = sup & (v.Y >= self.params["q"] * v.E)
            return np.where(ch, "change", "default"), ch | (v.N >= self.params["q"] * v.E)
        ok = T >= self.params["q"] * v.E
        return np.where(ok & sup, "change", "default"), ok


# ---------------------------------------------------------------- V8x combined limbs (v1.0.3)
class V8x(Rule):
    """Combined limbs (v1.0.3, for the March 2026 draft re-read in Section VII, tool run R8):
    a majority of valid votes (or a supermajority q_S), and optionally a participation limb
    T >= q_P E and an approval limb Y >= q_A E.  A valid default result needs the participation
    limb, N > Y and, where an approval limb exists, N >= q_A E (symmetric validity, Art. 118(3))."""
    code, name = "V8x", "combined limbs (participation, approval, majority)"
    symmetric_validity = True

    def __init__(self, q_P=None, q_A=None, q_S=None):
        super().__init__(**{k: v for k, v in dict(q_P=q_P, q_A=q_A, q_S=q_S).items() if v is not None})
        self.q_P, self.q_A, self.q_S = q_P, q_A, q_S
        self.default_monotone = q_P is None

    def _limbs(self, sx, sd, el):
        ivs = [iv_supermajority(self.q_S, sx, sd, el.n_ab, el.mu_A) if self.q_S else iv_majority(sx, sd, el.b_ab())]
        if self.q_P is not None:
            ivs.append(iv_participation(el.Q_participation(self.q_P), sx, sd))
        if self.q_A is not None:
            ivs.append(iv_approval(el.Q_approval(self.q_A), sx))
        return ivs

    def intervals(self, sx, sd, el):
        return [Piece(*inter(*self._limbs(sx, sd, el)))]

    def default_valid(self, sx, sd, el):
        mv, _ = iv_majority(sx, sd, el.b_ab())
        a, b = below(mv)
        if self.q_A is not None:
            sd_ = _arr(sd)
            Qd = self.q_A * el.rho - (1 - el.mu_A) * el.n_ab
            lim = np.where(sd_ > 0, 1.0 - Qd / np.where(sd_ > 0, sd_, 1.0), -INF)
            a, b = below(np.minimum(mv, lim))
        if self.q_P is not None:
            return inter((a, b), iv_participation(el.Q_participation(self.q_P), sx, sd))
        return a, b

    def evaluate(self, v):
        T = v.Y + v.N
        ok = np.ones_like(T, bool) if self.q_P is None else (T >= self.q_P * v.E)
        win = (v.Y >= self.q_S * T) & (T > 0) if self.q_S else (v.Y > v.N)
        ch = ok & win
        dv = ok & (v.N > v.Y)
        if self.q_A is not None:
            ch = ch & (v.Y >= self.q_A * v.E)
            dv = dv & (v.N >= self.q_A * v.E)
        return np.where(ch, "change", "default"), ch | dv


# ---------------------------------------------------------------- V9 graduated validity (approval form)
class V9(Rule):
    code, name = "V9", "graduated validity (void / advisory / binding)"
    symmetric_validity = True

    def __init__(self, qa=0.20, qb=1 / 3, lam=0.5, form="approval"):
        super().__init__(qa=qa, qb=qb, lam=lam, form=form)
        self.default_monotone = form == "approval"

    def _tier(self, q, sx, sd, el):
        if self.params["form"] == "approval":
            return inter(iv_majority(sx, sd, el.b_ab()), iv_approval(el.Q_approval(q), sx))
        return inter(iv_majority(sx, sd, el.b_ab()), iv_participation(el.Q_participation(q), sx, sd))

    def intervals(self, sx, sd, el):
        aa, ba = self._tier(self.params["qa"], sx, sd, el)
        ab, bb = self._tier(self.params["qb"], sx, sd, el)
        # advisory = lower tier minus the (nested) binding tier: the part below it and the part above it
        lo_a, lo_b = inter((aa, ba), (np.full_like(aa, -INF), np.minimum(ab, ba)))
        hi_a, hi_b = inter((aa, ba), (np.where(bb < INF, bb, INF), np.full_like(aa, INF)))
        lam = self.params["lam"]
        return [Piece(ab, bb, 1.0, "change"), Piece(lo_a, lo_b, lam, "advisory"), Piece(hi_a, hi_b, lam, "advisory")]

    def default_valid(self, sx, sd, el):
        sd = _arr(sd)
        Qd = self.params["qa"] * el.rho - (1 - el.mu_A) * el.n_ab
        lim = np.where(sd > 0, 1.0 - Qd / np.where(sd > 0, sd, 1.0), -INF)
        mv, _ = iv_majority(sx, sd, el.b_ab())
        return below(np.minimum(mv, lim))

    def evaluate(self, v):
        if self.params["form"] == "approval":
            m = v.Y
        else:
            m = v.Y + v.N
        maj = v.Y > v.N
        bind = maj & (m >= self.params["qb"] * v.E)
        adv = maj & (m >= self.params["qa"] * v.E) & ~bind
        out = np.where(bind, "change", np.where(adv, "advisory", "default"))
        return out, bind | adv | ((v.N > v.Y) & (v.N >= self.params["qa"] * v.E))


# ---------------------------------------------------------------- V10 double majority
class V10(Rule):
    code, name = "V10", "double majority (national majority and a majority in k_U of m_U units)"

    def __init__(self, k_U=7, m_U=12, sigma_u=0.05):
        super().__init__(k_U=k_U, m_U=m_U, sigma_u=sigma_u)

    def intervals(self, sx, sd, el):
        # Homogeneous units: the unit limb coincides with the national majority (heterogeneity is
        # handled in the Monte Carlo layer with unit-level support mu_u = mu + sigma_u * eps_u).
        a, b = iv_majority(sx, sd, el.b_ab())
        return [Piece(a, b)]

    def evaluate(self, v):
        nat = v.Y > v.N
        if v.units_Y is None:
            return np.where(nat, "change", "default"), v.Y != v.N
        wins = (v.units_Y > v.units_N).sum(axis=-1)
        ch = nat & (wins >= self.params["k_U"])
        return np.where(ch, "change", "default"), v.Y != v.N


# ---------------------------------------------------------------- V11 consultative status
class V11(Rule):
    code, name = "V11", "consultative (advisory) referendum carrying a validity rule"
    binding = False

    def __init__(self, base: Rule | None = None, lam=0.5):
        super().__init__(lam=lam)
        self.base = base or V1()
        self.default_monotone = self.base.default_monotone
        self.symmetric_validity = self.base.symmetric_validity

    def describe(self):
        return f"V11 consultative, lambda_adv={self.params['lam']:g}, carrying {self.base.describe()}"

    def intervals(self, sx, sd, el):
        return [Piece(p.a, p.b, p.weight * self.params["lam"], p.kind) for p in self.base.intervals(sx, sd, el)]

    def default_valid(self, sx, sd, el):
        return self.base.default_valid(sx, sd, el)

    def evaluate(self, v):
        return self.base.evaluate(v)


RULES = {"V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6, "V7": V7, "V8": V8,
         "V8x": V8x, "V9": V9, "V10": V10, "V11": V11}


def make_rule(spec) -> Rule:
    """Build a rule from a code string or a dict {'code': 'V3', 'q': 0.25, ...}."""
    if isinstance(spec, Rule):
        return spec
    if isinstance(spec, str):
        spec = {"code": spec}
    spec = dict(spec)
    code = spec.pop("code")
    if code in ("V5a", "V5p"):
        return V5(form="approval" if code == "V5a" else "participation", **spec)
    if code.startswith("V6") and len(code) == 3:
        return V6(variant=code[2], **spec)
    if code == "V11":
        base = spec.pop("base", "V1")
        return V11(base=make_rule(base), **spec)
    return RULES[code](**spec)


def menu(rho_full=1.934) -> list:
    """The validity menu at representative parameters (D6 ranges), for the full grid."""
    return [V1(), V2(q=0.25), V2(q=0.40), V2(q=0.50), V3(q=1 / 3), V3(q=0.25), V3(q=0.20),
            V5(qT=0.5, form="approval"), V5(qT=0.5, form="participation"),
            V6(q=1 / 3, variant="d"), V6(q=0.25, variant="d"),
            V7(qS=0.55), V7(qS=0.6), V7(qS=2 / 3),
            V8(q=1 / 3, qS=0.55), V8(q=0.25, qS=0.55),
            V9(qa=0.20, qb=1 / 3, lam=0.5), V10(k_U=7, m_U=12), V11(base=V1(), lam=0.5)]

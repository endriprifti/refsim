"""refsim v1.1.2: TR29, the indifference weight of V7 at 0.55 against V1 for T6 under accession priors (Appendix H)."""
import pytest

from refsim.design import weight_vs
from refsim.electorate import national
from refsim.evaluate import cell, scenario_settings
from refsim.params import Params
from refsim.priors import beta_shift
from refsim.rules import V1, V7


@pytest.fixture(scope="module")
def P():
    return Params.load("calibrated_2026")


def _w(P, mean, tau):
    st = scenario_settings(P)
    el = national(P, "E_full")
    pr = beta_shift(pi=1.0, a_B=st["a_B"], mu_b=mean)
    c = [cell(r, el, tau, 0.30, st["pi_by_type"]["T6"], 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau, P.behaviour.Theta,
              recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"], prior=pr) for r in (V1(), V7(qS=0.55))]
    w, how = weight_vs(dict(C1p_b=c[0]["C1p"], C1R_b=c[0]["C1R"]), dict(C1p_b=c[1]["C1p"], C1R_b=c[1]["C1R"]))
    return w, how, c


def test_mean_half_reproduces_TR22(P):
    # T6 = T1 under D13: the centre's w* of V7 at 0.55 is TR22's 3.30561 (digest §45)
    w, how, _ = _w(P, 0.50, 0.33)
    assert how == "preferred to V1 iff w > w*" and abs(w - 3.30561) < 1e-4


def test_accession_weights_rise_with_the_mean(P):
    ws = [_w(P, m, 0.33)[0] for m in (0.50, 0.60, 0.70, 0.80)]
    assert all(b > a for a, b in zip(ws, ws[1:]))
    assert abs(ws[1] - 3.717) < 1e-3 and abs(ws[2] - 4.045) < 1e-3 and abs(ws[3] - 4.115) < 1e-3


def test_cells_are_TR23s(P):
    _, _, c = _w(P, 0.70, 0.33)
    assert round(c[0]["C1p"], 3) == 0.011 and round(c[0]["C1R"], 3) == 0.013      # §46, Table VIII.2d
    assert round(c[1]["C1p"], 3) == 0.001 and round(c[1]["C1R"], 3) == 0.058

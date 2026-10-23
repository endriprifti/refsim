"""Regression tests against Section V (Parts 1 and 2) and Consistency check 2."""
import datetime as dt
import math

import pytest

from refsim import campaign, opinion, pipeline, regime_b, window
from refsim._legacy import load
from refsim.calendar import Calendar
from refsim.drift import lock_table, municipal_drift, pr_ever_falls
from refsim.electorate import adult_decline_from_instat, load_municipalities
from refsim.evaluate import t4_feasible_share
from refsim.priors import Prior, f_half_beta

RF, RD = 3713761 / 1.92e6, 3467826 / 1.92e6


def test_pipeline_ts0_prompt():
    r = pipeline.pr_vote("O0", "earliest")
    assert r["Pr_vote_12"] == pytest.approx(0.0607, abs=5e-4)
    assert r["Pr_vote_24"] == pytest.approx(0.9950, abs=5e-4)
    assert r["mean_months"] == pytest.approx(16.43, abs=0.01)


def test_pipeline_latest():
    assert pipeline.pr_vote("O0", "latest")["Pr_vote_24"] == pytest.approx(0.282, abs=2e-3)


def test_t4_prop23():
    assert t4_feasible_share(24) == 1.0
    assert t4_feasible_share(25) == pytest.approx(6 / 7)
    assert t4_feasible_share(31) == 0.0


def test_learning():
    assert campaign.L_star(124, 0.5, 0.8) == pytest.approx(76.8, abs=0.1)
    assert campaign.L_star(124, 1.0, 0.8) == pytest.approx(855.1, abs=0.1)
    assert campaign.gamma_c(10, 1.93 / 3) == pytest.approx(0.111, abs=1e-3)
    assert campaign.k_min(1.93 / 3, 0.0, 0.0, 1.0) == 9


def test_calendar_matches_legacy():
    L = load("v2", "calendar_al")
    C = Calendar()
    for y in range(2025, 2059):
        assert C.feas[y] == L.FEAS[y]


def test_repeat_dates_illustration():
    C = Calendar()
    fv = dt.date(2027, 5, 30)
    assert C.repeat_polling_day(fv, "BP") == dt.date(2030, 6, 2)
    assert C.repeat_polling_day(fv, "BF") == dt.date(2032, 1, 4)
    assert C.repeat_polling_day(fv, "VR") == dt.date(2029, 1, 7)


def test_prop25():
    for q, v in ((0.5, 4.328), (1 / math.e, 3.000), (0.25, 2.164), (0.10, 1.303)):
        assert opinion.critical_T_rel(3, q) == pytest.approx(v, abs=1e-3)


def test_prop27_first_passage():
    b = RF / 3
    assert opinion.mean_first_passage(1, 1, b) == pytest.approx(0.6836, abs=1e-3)
    assert opinion.mean_first_passage(2.5, 2.5, b) == pytest.approx(1.1468, abs=1e-3)
    assert opinion.d_first_passage_d_rho(1, 1, b) == pytest.approx(1.877, abs=2e-3)
    assert opinion.d_first_passage_d_rho(2.5, 2.5, b) == pytest.approx(3.595, abs=2e-3)


def test_corollary9_density():
    assert [round(f_half_beta(a), 3) for a in (0.5, 1, 4, 16)] == [0.637, 1.0, 2.188, 4.478]


def test_ochiai_granularity():
    assert opinion.ochiai({1, 2}, {1}) == pytest.approx(0.707, abs=1e-3)
    assert not opinion.same_issue({1, 2, 3, 4}, {1}, True)


def test_table_V24_and_bound():
    rows = window.pi_star_at_distance(RF, T_post=2.2)
    assert [round(p, 2) for _, p in rows] == [0.53, 0.37, 0.25, 0.17, 0.11, 0.06, 0.03, 0.01]
    assert window.two_point_bound(0.38) == pytest.approx(2.122, abs=1e-3)
    assert window.two_point_bound(0.38, t_25=0.403) == pytest.approx(2.132, abs=1e-3)
    assert window.two_point_bound(0.41) == float("inf")


def test_drift_years_fast_and_central():
    fast = [r["year"] for r in lock_table(3713761, 1.92e6, 31200, 0.0116)]
    central = [r["year"] for r in lock_table(3713761, 1.92e6, 31200, 0.0085)]
    assert fast == [2025, 2027, 2031, 2034, 2036, 2038, 2040]
    assert central == [2025, 2027, 2032, 2036, 2038, 2040, 2043]


def test_prop29_fall_probability():
    assert pr_ever_falls(RD, 1.7803, 0.020, 0.01) == pytest.approx(0.0031, abs=1e-4)
    assert pr_ever_falls(RD, 1.7803, 0.020, 0.02) == pytest.approx(0.2364, abs=1e-4)


@pytest.mark.parametrize("mp,n,printed,basis25,r1,y2,y5", [("2014", 61, 8, 9, 3, 12, 16),    # Law No. 115/2014, as printed
                                                         ("2027", 46, 6, 7, 2, 10, 12)])   # Law No. 100/2026 (v1.1.5)
def test_municipal_counts_D11(mp, n, printed, basis25, r1, y2, y5):
    m = load_municipalities(map=mp)
    assert len(m.names) == n
    assert m.count_above(R_now=None) == printed              # 2023-census basis
    assert m.count_above(R_now=1.92e6) == basis25            # 2025 basis (D11)
    assert len(m.remark1_exceptions(R_now=1.92e6)) == r1
    d = municipal_drift(31200 / 3713761 + 0.0085, map=mp)
    assert d["counts"][2] == y2 and d["counts"][5] == y5


def test_map_2027_aggregates_2014():
    """v1.1.5: the 46 municipalities of Law No. 100/2026 are whole-municipality mergers of the 61, within the same qark."""
    a, b = load_municipalities(map="2014"), load_municipalities(map="2027")
    assert abs(a.E_hat.sum() - b.E_hat.sum()) < 50 and abs(a.R_hat_2023.sum() - b.R_hat_2023.sum()) < 50
    qa, qb = a.qarks(1.92e6), b.qarks(1.92e6)
    assert sorted(qa) == sorted(qb) and all(abs(qa[k]["E"] - qb[k]["E"]) < 20 for k in qa)


def test_instat_D11():
    d = adult_decline_from_instat()
    assert d["d_total"] == pytest.approx(0.0116, abs=1e-4)
    assert d["d_15plus"] == pytest.approx(0.0083, abs=2e-4)


@pytest.mark.parametrize("den,rho,vals", [("E_dom", RD, (0.348, 0.339, 0.343)), ("E_full", RF, (0.428, 0.398, 0.383))])
def test_prop30_u_shaped(den, rho, vals):
    pr = Prior(0.5, 0.5)
    for tm, v in zip((0.0, 0.2, 0.35), vals):
        assert regime_b.tau_threshold_solver(pr, rho, tm) == pytest.approx(v, abs=1.5e-3)
        assert regime_b.tau_threshold_cap(pr, rho, tm) == pytest.approx(v, abs=1.5e-3)


@pytest.mark.parametrize("a", [1.0, 1.5, 2.5, 8.0])
@pytest.mark.parametrize("recal", [True, False])
def test_prop30_unimodal_dominance(a, recal):
    for rho in (1.5, RD, RF, 2.2):
        for tau in (0.3, 0.5):
            for tm in (0.0, 0.2, 0.35):
                if rho * tau > tm:
                    assert regime_b.dominance_check(Prior(a, a), rho, tm, tau, recalibrate=recal) <= 1e-12

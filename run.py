#!/usr/bin/env python3
"""Referendum Design Simulator — command line.

  python run.py reproduce        key values of Sections III-V recomputed (fast)
  python run.py packages         packages P0-P5 (central cell, 95% parameter intervals, local units)
  python run.py grid             the full menu over D3 x D9 x pi x denominator
  python run.py maps             regime maps over rho, tau, prior shape, drift rate, L and k, T_rel, window
  python run.py types            rules in force by type (T1-T6) over D3 x D9
  python run.py signatures       verification errors (C4') by rule, list size and margin
  python run.py pipeline         Pr_vote by timing scenario, presidential behaviour and verification rule
  python run.py convergence      Monte Carlo against exact quadrature (>= 10,000 replications per cell)
  python run.py sensitivity      sigma_tau sweep, noisy Pareto map, T4 weekday shares, literal national-register reading (T7, T8)
  python run.py cases            Section VII tool runs R1-R10 (historical cases, 2026 initiatives, draft re-read, comparators)
  python run.py calibrated       Section VII Part 2 reruns TR11-TR20 (use with --scenario calibrated_2026; refsim v1.1)
  python run.py design           Section VIII Part 1 tool runs TR21-TR28 and Appendix H's TR29 (use with --scenario calibrated_2026)
  python run.py accession        TR29 only: w* against V1 for T6 under accession priors (--means 0.6,0.7,0.8 or the author's mean)
  python run.py robustness       Section IX tool runs TR30-TR32 (use with --scenario calibrated_2026; refsim v1.1.3)
  python run.py local            v1.1.5: every run that uses the municipal map (local units, drift, literal reading, R10;
                                 with --scenario calibrated_2026: TR18, TR25, TR30(b)); see --map
  python run.py all              everything above

Options: --scenario NAME (config/NAME.yaml, default baseline_2025); --draws N (parameter-uncertainty draws);
--map 2027|2014 (v1.1.5): the municipal map, 46 municipalities of Law No. 100/2026 (default) or the 61 of Law No. 115/2014.
Outputs go to outputs/<scenario>/ (outputs/<scenario>_map2014/ with --map 2014) as CSV (+ .meta.json provenance) and figures.
"""
import argparse
import time

from refsim import runs
from refsim.params import Params

MODES = ["reproduce", "packages", "grid", "maps", "types", "signatures", "pipeline", "convergence", "sensitivity", "cases"]


def run_local(P, scen):
    """v1.1.5: the runs that depend on the municipal map."""
    if P.scenario.name.startswith("calibrated"):
        from refsim import calibrated, design, robustness
        calibrated.run_TR18(P, scen)
        design.run_TR25(P, scen)
        robustness.run_TR30(P, scen, parts=("b",))
    else:
        from refsim import cases
        runs.run_local_units(P, scen)
        runs.run_drift_maps(P, scen)
        runs.run_literal_reading(P, scen)
        cases.run_R10(P, scen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=MODES + ["calibrated", "design", "accession", "robustness", "local", "all"])
    ap.add_argument("--scenario", default="baseline_2025")
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--means", default="0.5,0.6,0.7,0.8", help="accession prior means for TR29 (comma-separated)")
    ap.add_argument("--map", default="2027", choices=["2027", "2014"],
                    help="municipal map: 46 municipalities of Law No. 100/2026 (2027) or 61 of Law No. 115/2014 (2014)")
    a = ap.parse_args()
    from refsim.electorate import set_municipal_map
    set_municipal_map(a.map)
    P = Params.load(a.scenario)
    scen = P.scenario.name + ("_map2014" if a.map == "2014" else "")
    todo = MODES if a.mode == "all" else [a.mode]
    for m in todo:
        t = time.time()
        if m == "packages":
            runs.run_packages(P, scen, n_draws=a.draws)
        elif m == "cases":
            from refsim import cases
            cases.run_all_cases(P, scen)
        elif m == "calibrated":
            from refsim import calibrated
            calibrated.run_all_calibrated(P, scen, n_draws=a.draws)
        elif m == "design":
            from refsim import design
            design.run_all_design(P, scen, n_draws=a.draws)
        elif m == "accession":
            from refsim import design
            design.run_TR29(P, scen, means=tuple(float(x) for x in a.means.split(",")))
        elif m == "robustness":
            from refsim import robustness
            robustness.run_all_robustness(P, scen)
        elif m == "local":
            run_local(P, scen)
        else:
            getattr(runs, f"run_{m}")(P, scen)
        print(f"{m}: done in {time.time() - t:.0f} s -> outputs/{scen}/")


if __name__ == "__main__":
    main()

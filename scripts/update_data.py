#!/usr/bin/env python3
"""Create a new scenario from new register, census and election figures.

Copies a base scenario (default: baseline_2025), replaces the values given on the command line,
stamps each replaced parameter with the new source and date, and writes config/<name>.yaml.
Optionally replaces the municipal table (Table C.4 layout) and appends election dates.

Examples
  python scripts/update_data.py --name lists_2027 --E-full 3780000 --E-ocv 262000 \
      --source "KQZ final lists, 2027 local elections" --date 2027-05-16
  python scripts/update_data.py --name census_2033 --R 1810000 --source "INSTAT 18+ residents" --date 2033-01-01
  python scripts/update_data.py --name lists_2027 --municipal data/my_table.csv --election 2029-05-13,2029-09-12,2029-09-14

Every value that is not given keeps its old source and date, so the provenance sidecar of each
output shows exactly which inputs are new.
"""
import argparse
import copy
import csv
import os
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIELDS = {  # command-line flag -> (section, key)
    "E_full": ("electorate", "E_full"), "E_ocv": ("electorate", "E_ocv"), "R": ("electorate", "R"),
    "T_dom": ("electorate", "T_dom_2025"), "tauR_ref": ("electorate", "tauR_ref"),
    "gamma_E": ("drift", "gamma_E"), "d_R_central": ("drift", "d_R_central"), "d_R_fast": ("drift", "d_R_fast"),
    "sigma_tau": ("behaviour", "sigma_tau"), "T_post": ("window", "T_post_months"), "T_rel": ("opinion", "T_rel_years"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="baseline_2025")
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", required=True, help="where the new figures come from")
    ap.add_argument("--date", required=True, help="date of the new figures (YYYY-MM-DD)")
    for f in FIELDS:
        ap.add_argument("--" + f.replace("_", "-"), dest=f, type=float)
    ap.add_argument("--municipal", help="CSV in the layout of data/table_c4_municipalities.csv")
    ap.add_argument("--election", action="append", default=[],
                    help="Assembly cycle 'polling_day,mandate_end,first_sitting' (observed); repeatable")
    a = ap.parse_args()

    base = yaml.safe_load(open(os.path.join(ROOT, "config", a.base + ".yaml"), encoding="utf-8"))
    new = copy.deepcopy(base)
    new["scenario"] = dict(name=a.name, description=f"{base['scenario']['description']} - updated: {a.source}",
                           data_vintage=a.date[:4])
    changed = []
    for f, (sec, key) in FIELDS.items():
        v = getattr(a, f)
        if v is None:
            continue
        leaf = new[sec][key]
        old = leaf["value"]
        leaf.update(value=int(v) if float(v).is_integer() and v > 1000 else v, source=a.source, date=a.date)
        changed.append(f"{sec}.{key}: {old} -> {leaf['value']}")
    out = os.path.join(ROOT, "config", a.name + ".yaml")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Scenario {a.name}: copy of {a.base} with new figures ({a.source}, {a.date})\n")
        yaml.safe_dump(new, fh, allow_unicode=True, sort_keys=False, width=140)
    if a.municipal:
        need = ["municipality", "qark", "census_pop_2023", "civil_registry_pop", "E_hat_2025", "R_hat_2023basis", "min_signatures"]
        hdr = next(csv.reader(open(a.municipal, encoding="utf-8")))
        if hdr != need:
            sys.exit(f"municipal CSV must have the columns {need}")
        shutil.copy(os.path.join(ROOT, "data", "table_c4_municipalities.csv"),
                    os.path.join(ROOT, "data", "table_c4_municipalities.previous.csv"))
        shutil.copy(a.municipal, os.path.join(ROOT, "data", "table_c4_municipalities.csv"))
        changed.append("data/table_c4_municipalities.csv replaced (previous kept as .previous.csv)")
    for e in a.election:
        with open(os.path.join(ROOT, "data", "assembly_cycles.csv"), "a", encoding="utf-8") as fh:
            fh.write(f"{e},observed,\"{a.source}\"\n")
        changed.append(f"assembly cycle added: {e}")
    print(f"wrote {out}")
    for c in changed:
        print("  " + c)
    print(f"next: python run.py reproduce --scenario {a.name}  (the baseline checks still use the printed 2025 values)\n"
          f"      python run.py packages --scenario {a.name}")


if __name__ == "__main__":
    main()

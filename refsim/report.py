"""Outputs: CSV files with a provenance sidecar, and the figures of Section VI (Figures 6.x).

Every CSV `outputs/<scenario>/<name>.csv` gets `<name>.meta.json` listing the scenario file, the data
vintage, the run time and every parameter value with its source and date.
Figures follow the reference palette (light mode): categorical slots blue / orange / aqua, a neutral gray
for "lock" and "no data", one-hue blue ramp for magnitudes; direct labels, recessive grid, one axis.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .params import ROOT  # noqa: E402

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#b9b8b2"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def out_dir(scenario: str) -> str:
    d = os.path.join(ROOT, "outputs", scenario)
    os.makedirs(d, exist_ok=True)
    return d


def write_csv(rows: list, name: str, params, scenario: str, note: str = "") -> str:
    d = out_dir(scenario)
    path = os.path.join(d, name + ".csv")
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})
    meta = dict(file=name + ".csv", scenario=params.path, data_vintage=params.scenario.data_vintage,
                run_time=dt.datetime.now().isoformat(timespec="seconds"), note=note, municipal_map=_map_label(),
                parameters=params.provenance())
    with open(os.path.join(d, name + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=str)
    return path


def _map_label():
    try:
        from .electorate import municipal_map, MUNICIPAL_MAPS
        return MUNICIPAL_MAPS[municipal_map()][1]
    except Exception:
        return None


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, name, scenario):
    d = out_dir(scenario)
    fig.patch.set_facecolor(SURFACE)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return os.path.join(d, name + ".png")

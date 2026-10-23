"""Calendar (Specification module 6).

National referendums (T1-T3) under Law 9087: no polling on an Assembly or local election day (119(1));
national window [t_end - 6 months, t_1 + 3 months] (119(2)); one polling day a year (Const. 152(3));
15 March cut-off (119(6)).  Cycles come from data/assembly_cycles.csv; later cycles follow A27
(t_end = previous first sitting + 4 years, t_1 = t_end + 2 days, polling on the second Sunday of May).
Variants: the polling-day-anchored window of F62 (start = Assembly polling day - `anchor_months`),
and the repeat calendar of the three-year bar (5.24) under H1 with the BP/BF switch (VR counterfactual).
Ported from legacy/section5_part2_B2/calendar_al.py; regression-tested against it.
"""
from __future__ import annotations

import csv
import datetime as dt
import os

from dateutil.relativedelta import relativedelta as rd

from .params import DATA_DIR

D = dt.date


def sunday_on_or_after(d: dt.date) -> dt.date:
    return d + dt.timedelta(days=(6 - d.weekday()) % 7)


def second_sunday_may(y: int) -> dt.date:
    return sunday_on_or_after(D(y, 5, 1)) + dt.timedelta(days=7)


def _date(s):
    return dt.date.fromisoformat(s)


class Calendar:
    def __init__(self, w_pre_months=6, w_post_months=3, anchor="mandate", anchor_months=4,
                 cutoff=(3, 15), first_year=2025, last_year=2061, cycles_csv=None, local_csv=None):
        self.w_pre, self.w_post = w_pre_months, w_post_months
        self.anchor, self.anchor_months = anchor, anchor_months
        self.cutoff = cutoff
        self.first_year, self.last_year = first_year, last_year
        rows = list(csv.DictReader(open(cycles_csv or os.path.join(DATA_DIR, "assembly_cycles.csv"), encoding="utf-8")))
        cyc = [(_date(r["polling_day"]), _date(r["mandate_end"]), _date(r["first_sitting"])) for r in rows]
        t1 = cyc[-1][2]
        y = cyc[-1][0].year + 4
        while y <= last_year + 4:
            t_end = t1 + rd(years=4)
            t_el = second_sunday_may(y)
            t1 = t_end + dt.timedelta(days=2)
            cyc.append((t_el, t_end, t1))
            y += 4
        self.cycles = cyc
        lrows = list(csv.DictReader(open(local_csv or os.path.join(DATA_DIR, "local_elections.csv"), encoding="utf-8")))
        local = [_date(r["polling_day"]) for r in lrows if r["kind"].startswith("general")]
        yl = max(local).year + 4
        while yl <= last_year + 4:
            local.append(second_sunday_may(yl))
            yl += 4
        self.local = local
        self.election_days = set([c[0] for c in cyc] + local)
        self.windows = [self._window(c) for c in cyc]
        self.feas = {y: self.feasible_sundays(y) for y in range(first_year, last_year + 1)}

    def _window(self, c):
        t_el, t_end, t1 = c
        start = (t_el - rd(months=self.anchor_months)) if self.anchor == "polling_day" else (t_end - rd(months=self.w_pre))
        return start, t1 + rd(months=self.w_post)

    def blocked(self, d: dt.date) -> bool:
        if d in self.election_days:
            return True
        return any(a <= d <= b for (a, b) in self.windows)

    def feasible_sundays(self, year: int):
        s = sunday_on_or_after(D(year, 1, 1))
        out = []
        while s.year == year:
            if not self.blocked(s):
                out.append(s)
            s += dt.timedelta(days=7)
        return out

    def blocked_share(self, y0=2026, y1=2041) -> float:
        tot = feas = 0
        for y in range(y0, y1 + 1):
            s = sunday_on_or_after(D(y, 1, 1))
            while s.year == y:
                tot += 1
                s += dt.timedelta(days=7)
            feas += len(self.feas[y])
        return 1 - feas / tot

    def first_feasible_on_or_after(self, d):
        for y in range(d.year, self.last_year + 1):
            for s in self.feas[y]:
                if s >= d:
                    return s
        return None

    def latest_in_year_on_or_after(self, d):
        for y in range(d.year, self.last_year + 1):
            c = [s for s in self.feas[y] if s >= d]
            if c:
                return c[-1]
        return None

    def target_year(self, clear: dt.date) -> int:
        return clear.year if (clear.month, clear.day) <= self.cutoff else clear.year + 1

    def poll_after_clearance(self, clear, g0_days=77, latest=False, not_before=None):
        y = self.target_year(clear)
        lb = max(clear + dt.timedelta(days=g0_days), D(y, 1, 1))
        if not_before is not None:
            lb = max(lb, not_before)
        return self.latest_in_year_on_or_after(lb) if latest else self.first_feasible_on_or_after(lb)

    # ---- the three-year bar on the calendar (Section V Part 2, (5.24), Corollary 8)
    def repeat_polling_day(self, first_vote: dt.date, reading="BP", collection_days=147, stages_days=200,
                           g0_days=77, latest=False, bar_years=3):
        """Repeat polling day of the same question.  H1: the bar runs from the earlier vote.
        BP: the bar constrains the repeat polling day (decree rule: not earlier than the anniversary);
        BF: the bar constrains the filing; VR: counterfactual bar keyed to a valid result (theta_const),
        filing four weeks after the void vote.  December completion deposits on 1 January (BF rule)."""
        anniv = first_vote + rd(years=bar_years)
        if reading == "BP":
            return self.first_feasible_on_or_after(anniv) if not latest else self.latest_in_year_on_or_after(anniv)
        start = anniv if reading == "BF" else first_vote + dt.timedelta(days=28)
        done = start + dt.timedelta(days=collection_days)
        if done.month == 12:
            done = D(done.year + 1, 1, 1)
        clear = done + dt.timedelta(days=stages_days)
        return self.poll_after_clearance(clear, g0_days, latest)


def months_between(a: dt.date, b: dt.date) -> float:
    return (b - a).days / 30.4375

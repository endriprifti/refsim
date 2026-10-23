"""Calendar for national (T1-T3) referendums under current law (A27).
Regular Assembly terms from the 2025 cycle; one polling day a year
(Const. 152(3)); 15 March cut-off (Code 2003, 119(6)); national window
119(2) = [t_end - 6 months, t_1 + 3 months]; no referendum on an
election day (119(1)).  Dates of 2029 onwards are assumptions."""
import datetime as dt
from dateutil.relativedelta import relativedelta as rd

D = dt.date

def sunday_on_or_after(d):
    return d + dt.timedelta(days=(6 - d.weekday()) % 7)

def second_sunday_may(y):
    return sunday_on_or_after(D(y, 5, 1)) + dt.timedelta(days=7)

# Assembly cycles: (t_el, t_end, t_1)
CYCLES = [(D(2021, 4, 25), D(2021, 9, 10), D(2021, 9, 10)),   # previous (t_end not used)
          (D(2025, 5, 11), D(2025, 9, 10), D(2025, 9, 12))]
t1 = D(2025, 9, 12)
for y in range(2029, 2058, 4):
    t_end = t1 + rd(years=4)
    t_el = D(2029, 5, 13) if y == 2029 else second_sunday_may(y)
    t1 = t_end + dt.timedelta(days=2)
    CYCLES.append((t_el, t_end, t1))

LOCAL = [D(2023, 5, 14), D(2027, 5, 16)] + [second_sunday_may(y) for y in range(2031, 2060, 4)]

WINDOWS = [(t_end - rd(months=6), t1 + rd(months=3)) for (_, t_end, t1) in CYCLES[1:]]
ELECTION_DAYS = set([c[0] for c in CYCLES] + LOCAL)

def blocked(d):
    if d in ELECTION_DAYS:
        return True
    return any(a <= d <= b for (a, b) in WINDOWS)

def feasible_sundays(year):
    s = sunday_on_or_after(D(year, 1, 1))
    out = []
    while s.year == year:
        if not blocked(s):
            out.append(s)
        s += dt.timedelta(days=7)
    return out

FEAS = {y: feasible_sundays(y) for y in range(2025, 2062)}

def first_feasible_on_or_after(d):
    y = d.year
    while y < 2062:
        for s in FEAS[y]:
            if s >= d:
                return s
        y += 1
    return None

def latest_in_year_on_or_after(d):
    y = d.year
    while y < 2062:
        c = [s for s in FEAS[y] if s >= d]
        if c:
            return c[-1]
        y += 1
    return None

CUTOFF = (3, 15)

def target_year(clear):
    return clear.year if (clear.month, clear.day) <= CUTOFF else clear.year + 1

def poll_after_clearance(clear, g0_days=77, latest=False, not_before=None):
    """Prompt President: earliest feasible Sunday >= clearance + g0 in the
    target year (rolling on if none); latest: last feasible Sunday of the
    target year that is >= clearance + g0."""
    y = target_year(clear)
    lb = max(clear + dt.timedelta(days=g0_days), D(y, 1, 1))
    if not_before is not None:
        lb = max(lb, not_before)
    return latest_in_year_on_or_after(lb) if latest else first_feasible_on_or_after(lb)

if __name__ == '__main__':
    for w in WINDOWS[:4]:
        print('window', w[0], w[1])
    print('blocked share 2025-2041 of Sundays: %.4f' %
          (1 - sum(len(FEAS[y]) for y in range(2026, 2042)) /
           sum(1 for y in range(2026, 2042) for k in range(53)
               if sunday_on_or_after(D(y, 1, 1)) + dt.timedelta(days=7 * k) < D(y + 1, 1, 1))))

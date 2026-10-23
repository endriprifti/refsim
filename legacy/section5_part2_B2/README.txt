Section V, Part 2 - code archive, Revision B2 (Appendix B, B.44-B.54)
Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, python-dateutil. Run each script from this folder.

Modules:   game.py, calendar_al.py, jacobi_fv.py (new in B2: exact finite-volume solver for the Jacobi diffusion)
V.3:       v3_voter.py, v3_exactac.py, v3_bar_calendar.py, v3_f23.py
V.4:       v4_lock.py, v4_peak.py (new), v4_passage.py, v4_exact.py (new; ~1 minute), v4_fv_check.py (new)
V.5:       v5_ou.py, v5_series.py
V.7:       v7_drift.py
Illustration: illustration.py
outputs/out_<script>.txt holds each script's output as run for Revision B2.

Changed in B2: v3_voter.py (granularity example, A2 literal r_v), v3_bar_calendar.py (BP/BF; December
deposit on 1 January; conditional median collection time), v3_f23.py (no-ballot cells), v4_passage.py
(Monte Carlo removed), v5_ou.py (Table V.23 w_post*, Remark 6 consistent bound and sensitivity,
Table V.24 with kappa at the stand-alone calibration, Proposition 12 check, local-election premium),
v7_drift.py (unrounded Table C.4 ratios, Remark 1 timing, joiners), illustration.py (VR/BP/BF labels,
exact transition law). Removed: v4_mc_check.py.

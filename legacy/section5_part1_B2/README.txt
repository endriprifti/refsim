Section V, Part 1 (Dynamic Layer I: pipeline, calendar and campaign) - code archive, Revision B2, 10 October 2026

Run order (Python 3.13, NumPy 2.5, SciPy 1.18):
  pipeline.py   module: chain, calendar, scheduler; B2 additions at the end (exact enumeration,
                no-foresight simulation, current-mandate calendar, whole-year losses)      -> Appendix B.38
  run_all.py    Tables V.2-V.4, V.12, B.40-1 (lambda_ee = 0)                 -> results.json     (B.39)
  run_b2.py     Tables V.5-V.7, Prop. 22 variants, Cor. 6(ii), indirect initiative,
                whole-year losses, clustered Monte Carlo errors, illustration -> results_b2.json  (B.43, about 3 min)
  gap.py, extra.py  decree-to-polling gaps, (5.7) derivatives, decomposition -> gap.json, extra.json (B.40)
  learning.py   Tables V.8-V.11, infeasibility thresholds, cue-taking, 1994   -> learning.json    (B.41)

Seeds: Tables V.5-V.6 at lambda_ee = 0.056 seed 5 (100,000 runs per cell, calendar redrawn every 20 runs);
slot counts seed 11; whole-year losses seed 21; Table V.4 seeds 7 and 9. Values at lambda_ee = 0 are exact.
Code labels: O0-O9 = TS0-TS9; president='earliest' = prompt President; LT1-LT4 = local timing readings.
All parameters marked illustrative in the text are illustrative here.

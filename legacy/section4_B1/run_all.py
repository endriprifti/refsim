"""Section IV: run every computation and write results_section4.txt (about 10 minutes)."""
import subprocess, sys, time
steps = [['s4_core.py'], ['s4_units.py'], ['s4_table1.py'], ['s4_beta.py'], ['s4_pollbeta.py'], ['s4_unitlock.py'],
         ['s4_bounds.py'], ['s4_trajectory.py'], ['s4_register.py'], ['s4_verify.py', 'base'], ['s4_verify.py', 'ext'],
         ['s4_margins.py'], ['s4_margins2.py'], ['s4_illustration.py']]
with open('results_section4.txt', 'w') as out:
    for s in steps:
        t = time.time(); out.write(f'\n===== {" ".join(s)} =====\n'); out.flush()
        r = subprocess.run([sys.executable] + s, capture_output=True, text=True)
        out.write(r.stdout + (('\nSTDERR:\n' + r.stderr) if r.returncode else '') + f'\n[{time.time()-t:.0f}s]\n'); out.flush()

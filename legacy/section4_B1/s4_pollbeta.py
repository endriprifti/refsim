"""IV.1(c): the poll model of Proposition 10 under uniform and Beta priors (Table IV.4, Table B.24-1)."""
import numpy as np
from s4_core import *
from s4_beta import prior_of
def sd_of(fam, pi, conc):
    return (1-pi)/np.sqrt(12*conc)
def run(rho, tau, tm, fam, pi, conc, H0):
    pr = prior_of(fam, pi, conc); h = H0*np.sqrt(3)*sd_of(fam, pi, conc)
    return poll_model(rho/3, rho*tau, tm, pr, h)
if __name__ == '__main__':
    rho = 1.93
    for H0 in (0.5, 0.25):
        print('H0', H0)
        for tau in (.25, .30, .40, .50):
            for tm in (0, .2, .35):
                cells = []
                for (f, pi, c) in [('U',0,1),('B',0,2),('U',.3,1),('B',.3,1),('B',.3,2)]:
                    pc, pcamp = run(rho, tau, tm, f, pi, c, H0); cells.append(f'{pc:.3f}/{pcamp:.2f}')
                print(tau, tm, '  '.join(cells))

"""Remark 6: local/Assembly turnout ratios by lag (Section II, Table C.3)."""
import datetime as dt
D = dt.date
asm = [(D(1991,3,31),98.6),(D(1992,3,22),90.4),(D(1996,5,26),89.1),(D(1997,6,29),72.6),
       (D(2001,6,24),53.6),(D(2005,7,3),48.0),(D(2009,6,28),50.8),(D(2013,6,23),53.5),
       (D(2017,6,25),46.8),(D(2021,4,25),46.3),(D(2025,5,11),44.8)]
loc = [(D(1992,7,26),70.7,''),(D(1996,10,20),58.5,''),(D(2000,10,1),50.7,''),(D(2003,10,12),45.7,''),
       (D(2007,2,18),46.3,''),(D(2011,5,8),50.9,''),(D(2015,6,21),47.3,''),(D(2019,6,30),22.97,'boycott'),
       (D(2023,5,14),38.23,''),(D(2025,11,9),18.45,'partial, 5 municipalities')]
for d, t, note in loc:
    prev = max(a for a in asm if a[0] < d)
    lag = (d - prev[0]).days / 30.4375
    print('%s local %.2f / Assembly %s %.1f = %.3f  lag %.1f months %s' % (d, t, prev[0], prev[1], t / prev[1], lag, note))

"""Table C.4 municipal proxies (Section II): name, qark, census 2023 C_u, civil-registry CR_u."""
import numpy as np
LAM_E, LAM_R = 0.8545, 0.8035
ROWS = """Tiranë,Tiranë,598176,757361
Durrës,Durrës,153614,299989
Elbasan,Elbasan,115101,205892
Shkodër,Shkodër,102434,200889
Fier,Fier,101963,196324
Vlorë,Vlorë,83683,194147
Korçë,Korçë,60754,129065
Lushnjë,Fier,63135,127438
Kamëz,Tiranë,96137,125632
Lezhë,Lezhë,51354,106245
Berat,Berat,62232,98875
Pogradec,Korçë,46070,90976
Krujë,Durrës,51191,79608
Kavajë,Tiranë,30012,79556
Dibër,Dibër,50775,78940
Kurbin,Lezhë,34405,72987
Maliq,Korçë,31008,64664
Kukës,Kukës,36125,59393
Kuçovë,Berat,31077,55293
Malësi e Madhe,Shkodër,21684,53918
Divjakë,Fier,24882,53372
Gjirokastër,Gjirokastër,23270,52054
Sarandë,Vlorë,22613,50680
Vau i Dejës,Shkodër,19261,48966
Cërrik,Elbasan,25163,46652
Librazhd,Elbasan,23312,44181
Shijak,Durrës,22058,44103
Mallakastër,Fier,15838,43033
Patos,Fier,18227,42738
Devoll,Korçë,25897,42388
Rrogozhinë,Tiranë,12567,40684
Bulqizë,Dibër,26826,40223
Dimal,Berat,28135,39551
Finiq,Vlorë,11413,39055
Peqin,Elbasan,16580,38900
Mat,Dibër,17405,38615
Mirditë,Lezhë,13625,37384
Gramsh,Elbasan,16533,36305
Vorë,Tiranë,21621,36230
Selenicë,Vlorë,9580,35793
Prrenjas,Elbasan,18768,33031
Belsh,Elbasan,17123,32389
Roskovec,Fier,16332,31548
Tropojë,Kukës,14189,28216
Himarë,Vlorë,8328,27049
Dropull,Gjirokastër,8259,23131
Memaliaj,Gjirokastër,6578,21467
Has,Kukës,11684,21247
Përmet,Gjirokastër,7980,20301
Kolonjë,Korçë,7519,19919
Skrapar,Berat,10750,19072
Poliçan,Berat,8762,18341
Delvinë,Vlorë,6166,18078
Konispol,Vlorë,4898,17737
Pukë,Shkodër,6222,16836
Klos,Dibër,12172,16618
Tepelenë,Gjirokastër,6761,16371
Këlcyrë,Gjirokastër,4400,12468
Fushë-Arrëz,Shkodër,4878,11651
Libohovë,Gjirokastër,2765,7158
Pustec,Korçë,1843,5191"""
units = []
for line in ROWS.split('\n'):
    n, q, c, cr = line.split(','); c, cr = int(c), int(cr)
    E = LAM_E*cr; R = LAM_R*c
    units.append(dict(name=n, qark=q, C=c, CR=cr, E=E, R=R, rho=E/R))
QARKS = sorted(set(u['qark'] for u in units))
def qark_table():
    out = []
    for q in QARKS:
        E = sum(u['E'] for u in units if u['qark']==q); R = sum(u['R'] for u in units if u['qark']==q)
        out.append(dict(qark=q, E=E, R=R, rho=E/R))
    return out
if __name__ == '__main__':
    import statistics
    print(len(units), 'median rho', round(statistics.median(u['rho'] for u in units),3))
    print('rho>3:', [u['name'] for u in units if u['rho']>3])
    print('rho<1.5:', [u['name'] for u in units if u['rho']<1.5])
    for r in qark_table(): print(r['qark'], round(r['E']), round(r['R']), round(r['rho'],3))
    for nm in ('Tiranë','Korçë','Pustec'):
        u=[x for x in units if x['name']==nm][0]; print(nm, round(u['E']), round(u['R']), round(u['rho'],2))

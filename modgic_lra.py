import sys
import numpy as np
from estampes.parser import DataFile, build_qlabel
from estampes.tools.atom import convert_labsymb
from estampes.data.physics import PHYSFACT
import logparse_copy as lp

tmpl = """%mem=4GB
%nprocshared=4
#P HF/3-21G geom=(Modredundant,GIC)

TEMPLATE

0 1
{CRD}
{GIC}


"""

def read_geom(fname):
    dkeys = {
        #    'Energy': build_qlabel(1),
            'atcrd': build_qlabel('atcrd', 'last'),
            'atnum': build_qlabel('atnum'),
            }
    
    dfile = DataFile(fname)
    data = dfile.get_data(*dkeys.values())
    #print(data)
    #atlab = np.array(data[dkeys['atoms']]['data']
    #atnum = np.array(data[dkeys['atnum']]['data'])
    atlab = convert_labsymb(True, *data[dkeys['atnum']]['data'])
    atcrd = np.array(data[dkeys['atcrd']]['data'])*PHYSFACT.bohr2ang
    #print(atlab)
    return (atlab, atcrd)


fname = sys.argv[1]
# level: B2VTZ, rDSDT, PW6D3
level = sys.argv[2]

datam = lp.read_molinfo(fname)
mol = read_geom(fname)
#data.fragments = [list(range(15)),]

tmpl_line = '{b:12s}{a[0]:15.6f}{a[1]:15.6f}{a[2]:15.6f}\n'
geom = ""
for j, lb in enumerate(mol[0]):
    geom += tmpl_line.format(a=mol[1][j],b=lb)


datam.lrafragred(0, 'R', level)
#datam.tplfragred(0, 'R', 'B2VTZ')
gic = lp.printmod_gic(datam)

fout = fname[:-4]+"_mod.com"
with open(fout, 'w') as fopen:
    fopen.write(tmpl.format(CRD=geom, GIC=gic))


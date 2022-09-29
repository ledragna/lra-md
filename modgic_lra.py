import sys
import numpy as np
import argparse
import logparse_copy as lp

tmpl = """%mem=4GB
%nprocshared=4
#P HF/3-21G geom=(Modredundant,GIC)

TEMPLATE

0 1
{CRD}
{GIC}


"""

def build_parser():
    """Builds options parser.
    Builds the full option parser.
    Returns
    -------
    :obj:`ArgumentParser`
        `ArgumentParser` object
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('fname', type=str, help="The gaussian log file")
    parser.add_argument('-l', '--level', type=str, choices=['B2VTZ', 'rDSDT',
                                                            'PW6D3'],
                        help="level of theory", default='rDSDT')
    parser.add_argument('--template', action="store_true", 
                        help="Add template corraction to the available bond type")
    return parser


def main():
    par = build_parser()
    opts = par.parse_args()
    fname = opts.fname
    # levels: B2VTZ, rDSDT, PW6D3
    level = opts.level

    datam = lp.read_molinfo(fname)

    tmpl_line = '{b:12s}{a[0]:15.6f}{a[1]:15.6f}{a[2]:15.6f}\n'
    geom = ""
    for j, lb in enumerate(datam.atlabs):
        geom += tmpl_line.format(a=datam.atmcrd[j,:], b=lb)


    datam.lrafragred(0, 'R', level)
    if opts.template:
        datam.tplfragred(0, 'R', level)
    gic = lp.printmod_gic(datam)

    fout = fname[:-4]+"_mod.com"
    with open(fout, 'w') as fopen:
        fopen.write(tmpl.format(CRD=geom, GIC=gic))

if __name__ == '__main__':
    main()

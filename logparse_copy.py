#!/usr/bin/env python
import re
import copy
import numpy as np
from typing import TypedDict, List, Dict
from estampes.parser import DataFile, build_qlabel
from estampes.tools.atom import convert_labsymb
from estampes.data.physics import PHYSFACT
import calculate_rmsd as rmsd

class IntCrd(TypedDict):
    cordef: List[int]
    corval: float

class ModCrd(TypedDict):
    R: List[IntCrd]
    A: List[IntCrd]
    L: List[IntCrd]
    D: List[IntCrd]

#IntCrd = Dict(str, ModCrd)
#StrVec = List[str]
#IntVecVec = List[List[int]]

LRA_PARAM = { 'B2VTZ': {'CH': [-0.074795+1,  0.080798],
                        'HO': [-0.289543+1,  0.276062],
                        'CC': [-0.015977+1,  0.022342],
                        'HN': [ 0.014185+1, -0.015295],
                        'CO': [-0.004834+1,  0.003455],
                        'CF': [-0.000953+1, -0.001231],
                        'CN': [ 0.006962+1, -0.010544],
                        'CCl':[-0.108290+1,  0.174512]},
              'rDSDT': {'CH': [-0.00239+1, 0],
                        'HO': [0.24674 +1, -0.24091],
                        'CC': [-0.00184+1, 0],
                        'HN': [-0.00216+1, 0],
                        'CO': [-0.00297+1, 0],
                        'CF': [-0.00307+1, 0 ],
                        'CN': [-0.00234+1, 0],
                        'CS': [-0.01222+1, 0.01672],
                        'CCl': [-0.0043+1, 0]},
              'PW6D3': {'CH': [-0.00586+1, 0],
                        'HO': [0.17529 +1, -0.17005],
                        'CC': [0.00014 +1, 0],
                        'HN': [-0.00331+1, 0],
                        'CO': [0.01708+1, -0.0212],
                        'CF': [-0.00598+1, 0],
                        'CN': [0.01705 +1, -0.02079],
                        'CS': [-0.01296+1, 0.02188],
                        'CCl': [-0.0014+1, 0]},

            }

TPL_PARAM = { 'B2VTZ': {'NO': -0.00484515},
              'rDSDT': {'SH': 0.00299}}

class KeywordError(Exception):
    """Generates an error if keyword not found.
    """
    def __init__(self, name, msg=None):
        if msg is None:
            msg = f'Unrecognized keyword: {name}'
        super(KeywordError, self).__init__(msg)


def islistempty(inlist):
    if isinstance(inlist, list): # Is a list
        return all( map(islistempty, inlist) )
    return False

def calc_rvecmat(coord):
    """read a set of atom cordinates (NATx3)
    and return a distance vector matrix

    Arguments:
        coord {np.array(natoms,3)} -- Cartesian coordinates

    Returns:
        np.array(natoms,natoms,3) -- [description]
    """
    rvec = coord[:,np.newaxis,:] - coord[np.newaxis,:,:]
    return rvec


def distance_mat(rvec):
    """ reads the distance vector matrix and
    returns the euclidean distance matrix

    Arguments:
        rvec {np.array(natoms,natoms,3)} -- distance vector matrix

    Returns:
        np.array(natoms,natoms) -- euclidean distance matrix
    """
    dist = np.sqrt(np.einsum('ijk,ijk->ij', rvec, rvec))
    return dist


def angle(rvec, ith, jth, kth):
    """
    Reads distance vector matrix and three atom indices
    and returns the angle between them, the secodn one is the vertex

    Arguments:
        rvec {[type]} -- distance vector matrix
        ith {[type]} -- first atom index
        jth {[type]} -- second atom index (angle vertex)
        kth {[type]} -- third atom index

    Returns:
        [float] -- angle as degree
    """
    cos_alpha = np.dot(rvec[ith, jth], rvec[kth, jth])
    sin_alpha = np.linalg.norm(np.cross(rvec[ith, jth], rvec[kth, jth]))
    alpha = np.arctan2(sin_alpha, cos_alpha)
    return np.degrees(alpha)


def angle_plane(coord, centr, pl1, pl2, out):
    """
    compute the out of plane angle of out
    """
    vec1 = coord[pl1, :]- coord[centr, :]
    vec2 = coord[pl2, :]- coord[centr, :]
    veco = coord[out, :]- coord[centr, :]
    vert = np.cross(vec1,vec2)
    cos_alpha = np.dot(vert, veco)
    sin_alpha = np.linalg.norm(np.cross(vert, veco))
    alpha = np.arctan2(sin_alpha, cos_alpha)
    return 90. - np.degrees(alpha)


# https://stackoverflow.com/a/34245697
def dihedral(rvec, ith, jth, kth, lth):
    """Praxeolitic formula 1 sqrt, 1 cross product
    taken from https://stackoverflow.com/a/34245697

    Arguments:
        rvec {np.array(natoms,natoms,3)} -- [description]
        ith {int} -- first atom index
        jth {int} -- second atom index
        kth {int} -- third atom index
        lth {int} -- fourth atom index

    Returns:
        float -- dihedral angle in degree
    """

    b0 = rvec[ith, jth]  # -1.0*(p1 - p0)
    b1 = rvec[kth, jth]  # p2 - p1
    b2 = rvec[lth, kth]  # p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.sqrt(np.einsum('...i,...i', b1, b1))  # np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

class IntCoordinate():
    """
    super class for internal coordinates
    """
    def __init__(self, indx, value, typo):
        self.indx = indx
        self.value = value
        self.tps = typo

    def __repr__(self):
        toprt = self.tps.__repr__() 
        toprt += ' '
        toprt += self.indx.__repr__()
        toprt += ' '
        toprt += self.value.__repr__()
        return toprt

    def _samecrd(self, other):
        if self.indx == other.indx and self.tps == other.tps:
            return True
        else:
            # FIXME use an exception
            return False

    def _checktypeop(self, other):
        if isinstance(other, IntCoordinate):
            if self._samecrd(other):
                value = other.value
            else:
                print('WARNING different RedCoordinate')
                return None
        elif isinstance(other, (float, int)):
            value = other
        else:
            print('wtf?')
            return None
        return value

    def __add__(self, other):
        if isinstance(other, IntCoordinate):
            value = self._checktypeop(other)
        else:
            value = other
        return IntCoordinate(self.indx, self.value + value, self.tps)

    def __sub__(self, other):
        if isinstance(other, IntCoordinate):
            value = self._checktypeop(other)
        else:
            value = other
        return IntCoordinate(self.indx, self.value - value, self.tps)

    def __mul__(self, other):
        if isinstance(other, IntCoordinate):
            value = self._checktypeop(other)
        else:
            value = other
        return IntCoordinate(self.indx, self.value * value, self.tps)

    def __truediv__(self, other):
        if isinstance(other, IntCoordinate):
            value = self._checktypeop(other)
        else:
            value = other
        return IntCoordinate(self.indx, self.value / value, self.tps)


class SetIntCoord():
    # FIXME just a shitty class to try the stuff
    def __init__(self, values):
        self.vals = values
        self._set_pointer()

    def __repr__(self):
        toprint = ''
        tmplstr = ' {:1s}{:<10s} {:10.6f}\n'
        for key in self.vals.keys():
            for valz in self.vals[key]:
                indrep = '('
                for ind in valz.indx:
                    indrep += '{:d},'.format(ind)
                indrep = indrep[:-1]+')'
                toprint += tmplstr.format(key, indrep, valz.value)
        return toprint

    def _set_pointer(self):
        tmp_dic = {}
        mod_dic = {}
        for key in self.vals.keys():
            tmp_dic[key] = {}
            mod_dic[key] = {}
            for ith, valz in enumerate(self.vals[key]):
                kname = key+'{:04d}'*len(valz.indx)
                kname = kname.format(*valz.indx)
                tmp_dic[key][kname] = ith
                mod_dic[key][kname] = False
        self._refdic = tmp_dic
        self._moddic = mod_dic

    def __add__(self, other):
        # FIXME no checks at all
        try:
            tmp_set = copy.deepcopy(self)
            for key in self.vals.keys():
                for ith in range(len(tmp_set.vals[key])):
                    tmp_set.vals[key][ith].value += other.vals[key][ith].value
            return tmp_set
        except:
            print('different size, nothing done')

    def __sub__(self, other):
        # FIXME no checks at all
        try:
            tmp_set = copy.deepcopy(self)
            for key in self.vals.keys():
                for ith in range(len(tmp_set.vals[key])):
                    tmp_set.vals[key][ith].value -= other.vals[key][ith].value
            return tmp_set
        except:
            print('different size, nothing done')

    def addmap(self, other, kmap):
        tmp_set = copy.deepcopy(self)
        for key in other.vals.keys():
            for valz in other.vals[key]:
                if np.abs(valz.value) > 1e-5:
                    kname = key+'{:04d}'*len(valz.indx)
                    nind = [kmap[x] for x in valz.indx]
                    kname = kname.format(*nind)
                    newith = self._refdic[key][kname]
                    tmp_set.vals[key][newith].value += valz.value
                    tmp_set._moddic[key][kname] = True
        return tmp_set

    def replacemap(self, other, kmap):
        tmp_set = copy.deepcopy(self)
        for key in other.vals.keys():
            for valz in other.vals[key]:
                if np.abs(valz.value) > 1e-5:
                    kname = key+'{:04d}'*len(valz.indx)
                    nind = [kmap[x] for x in valz.indx]
                    kname = kname.format(*nind)
                    newith = self._refdic[key][kname]
                    tmp_set.vals[key][newith].value = valz.value
                    tmp_set._moddic[key][kname] = True
        return tmp_set

    def get_mod(self):
        tmp_dic = {}
        for key in self._moddic.keys():
            tmp_dic[key] = []
            for key2 in self._moddic[key]:
                if self._moddic[key][key2]:
                    tmp_dic[key].append(self._refdic[key][key2])
            if not tmp_dic[key]:
                del tmp_dic[key]

        return tmp_dic

    def lracorrect(self, typo, level, kmap):
        """
        Finire da finire!!!
        """
        #pass
        tmp_set = copy.deepcopy(self)
        for valz in self.vals[typo]:
            kname = typo+'{:04d}'*len(valz.indx)
            nind = [kmap[x] for x in valz.indx]
            kname = kname.format(*nind)
            newith = self._refdic[typo][kname]
            _tps = tmp_set.vals[typo][newith].tps
            if _tps in LRA_PARAM[level]:
                apar, bpar = LRA_PARAM[level][_tps]
                tmp_set.vals[typo][newith].value = valz.value * apar + bpar
                tmp_set._moddic[typo][kname] = True
        return tmp_set

    def tplcorrect(self, typo, level, kmap):
        """
        Finire da finire!!!
        """
        #pass
        tmp_set = copy.deepcopy(self)
        for valz in self.vals[typo]:
            kname = typo+'{:04d}'*len(valz.indx)
            nind = [kmap[x] for x in valz.indx]
            kname = kname.format(*nind)
            newith = self._refdic[typo][kname]
            _tps = tmp_set.vals[typo][newith].tps
            if _tps in TPL_PARAM[level]:
                apar = TPL_PARAM[level][_tps]
                tmp_set.vals[typo][newith].value = valz.value + apar
                tmp_set._moddic[typo][kname] = True
        return tmp_set



def getredcoord(fname):
    """Read a gaussian log file and get the redundant internal coordinates


    Arguments:
      fname {string} -- Gaussian log file

    Returns:
        dict -- dictionary of the data {atoms:[]}
    """
    keyint1 = '!   Optimized Parameters   !'
    keyint2 = '!    Current Parameters    !'
    #                           ----------------------------
    #                           !   Optimized Parameters   !
    #                           ! (Angstroms and Degrees)  !
    # --------------------------                            --------------------------
    # ! Name  Definition              Value          Derivative Info.                !
    #--------------------------------------------------------------------------------
    res = {'R': [],
           'A': [],
           'L': [],
           'D': []}
    with open(fname, 'r') as fopen:
        line = fopen.readline()
        while (keyint1 not in line) and (keyint2 not in line):
            line = fopen.readline()
            if not line:
                raise KeywordError('Frequencies')
        # Skipt 5 line
        for _ in range(5):
            line = fopen.readline()
        tokens = line.split()
        while len(tokens) > 1:
            crdind = re.split('\\(|,|\\)', tokens[2])
            if crdind[0] in res.keys():
                res[crdind[0]].append(IntCoordinate([int(x)-1 if int(x) > 0 else int(x) for x in crdind[1:-1]],
                float(tokens[3]), None))
            line = fopen.readline()
            tokens = line.split()
    return res

def read_molinfo(fname):
    """

    Arguments:
        fname {[type]} -- [description]
    """
    dkeys = {
        #    'Energy': build_qlabel(1),
            'atcrd': build_qlabel('atcrd', 'last'),
            'atnum': build_qlabel('atnum'),
            }
    dfile = DataFile(fname)
    data = dfile.get_data(*dkeys.values())
    atnum = np.array(data[dkeys['atnum']]['data'])
    #atlab = convert_labsymb(True, *data[dkeys['atnum']]['data'])
    #print(atlab)
    atcrd = np.array(data[dkeys['atcrd']]['data'])*PHYSFACT.bohr2ang

    intcrd = SetIntCoord(getredcoord(fname))
    return MolCoord(atnum, atcrd, intcrd)

class MolCoord():
    """Class to contain the redundant interna coordinates of a molecule
    and manipulate them
    """
    def __init__(self, atoms, atmcrd, intcrd, fragments=None):
        """

        Arguments:
            atoms {[type]} -- [description]
            atmcrd {[type]} -- [description]
            intcrd {[type]} -- [description]

        Keyword Arguments:
            fragments {list} -- [description] (default: {[[]]})
        """
        self.atoms = atoms
        self.atlabs = convert_labsymb(True, *atoms.tolist())
        self.atmcrd = atmcrd
        self.intcrd = intcrd
        self._updateintcrdtype()
        self.modintcrd = copy.deepcopy(intcrd)
        if fragments is None:
            fragments = [list(range(self.atoms.shape[0]))]
        self.fragments = fragments



    @property
    def atoms(self):
        """
        Return the name of the molecular system
        """
        return self.__atoms

    @atoms.setter
    def atoms(self, val):
        """Sets the name of the molecular system

        Arguments:
            val {str} -- Atom labels list
        """
        self.__atoms = val

    @property
    def atmcrd(self):
        """
        Return the name of the molecular system
        """
        return self.__atmcrd

    @atmcrd.setter
    def atmcrd(self, val):
        """Sets the name of the molecular system

        Arguments:
            val {str} -- Atom labels list
        """
        self.__atmcrd = val
        self.__rvec = calc_rvecmat(val)
        self.__dmat = distance_mat(self.__rvec)

    @property
    def intcrd(self):
        """
        Return the name of the molecular system
        """
        return self.__intcrd

    @intcrd.setter
    def intcrd(self, val):
        """Sets the name of the molecular system

        Arguments:
            val {str} -- Atom labels list
        """
        self.__intcrd = val
        self._updateintcrdtype()

    @property
    def fragments(self):
        """
        Return the name of the molecular system
        """
        return self.__fragments

    @fragments.setter
    def fragments(self, val):
        if islistempty(val):
            val = []
        tmp_full = [item for sublist in val for item in sublist]
        mol_atoms = self.atoms.shape[0]
        natms = len(tmp_full)
        unic_atms = set(tmp_full)
        lunic_atms = len(unic_atms)
        if natms != lunic_atms:
            print("WARNING: some atoms present more than one fragments")
        try:
            unic_atms = np.array(list(unic_atms))
            if (unic_atms < 0).any() or (unic_atms >= mol_atoms).any():
                self.__frag = None
                raise ValueError
            else:
                excluded = list(set(range(mol_atoms)) - set(tmp_full))
                if excluded:
                    val.append(excluded)
                self.__fragments = [sorted(x) for x in val]

        except Exception as err: print(err)

    def getfragcrd(self, fragind):
        if not self.__fragments:
            raise KeywordError('fragments', msg='Fragments not setted')
        if int(fragind) > len(self.__fragments):
            raise KeywordError('fragmentsindx', msg='Fragment undefined max val:{}'.format(len(self.__fragments-1)))

        return self.__atmcrd[self.__fragments[fragind]]

    def getfragatm(self, fragind):
        if not self.__fragments:
            raise KeywordError('fragments', msg='Fragments not setted')
        if int(fragind) > len(self.__fragments):
            raise KeywordError('fragmentsindx', msg='Fragment undefined max val:{}'.format(len(self.__fragments-1)))

        return self.__atoms[self.__fragments[fragind]]

    def getfragred(self, fragind):
        if not self.__fragments:
            raise KeywordError('fragments', msg='Fragments not setted')
        if int(fragind) > len(self.__fragments):
            raise KeywordError('fragmentsindx', msg='Fragment undefined max val:{}'.format(len(self.__fragments-1)))

        indmap = {x:y for y,x in enumerate(self.__fragments[fragind])}
        res = {}
        for key in self.__intcrd.vals.keys():
            # FIXME Linear coordinate discharged
            if key != 'L':
                res[key] = []
                for valz in self.__intcrd.vals[key]:
                    if all(elem in self.__fragments[fragind] for elem in valz.indx):
                        res[key].append(IntCoordinate([indmap[x] for x in
                                                       valz.indx], valz.value,
                                                     valz.tps))
                if not res[key]:
                    del res[key]
        return res

    def getfragmol(self, fragid):
        """
        ...
        """
        return MolCoord(self.getfragatm(fragid), self.getfragcrd(fragid),
                        self.getfragred(fragid))

    def modfragred(self, fragid, redmod):
        """
        add a delta to all the defined gic
        """
        indmap = dict(enumerate(self.__fragments[fragid]))
        self.modintcrd = self.modintcrd.addmap(redmod, indmap)

    def replacefragred(self, fragid, redmod):
        """
        add a delta to all the defined gic
        """
        indmap = dict(enumerate(self.__fragments[fragid]))
        self.modintcrd = self.modintcrd.replacemap(redmod, indmap)

    def lrafragred(self, fragid, typo, level):
        """
        corrects the coordinates of type (typo: R, A, D) as y = ax + b
        """
        indmap = dict(enumerate(self.__fragments[fragid]))
        self.modintcrd = self.modintcrd.lracorrect(typo, level, indmap)

    def tplfragred(self, fragid, typo, level):
        """
        corrects the coordinates of type (typo: R, A, D) as y = ax + b
        """
        indmap = dict(enumerate(self.__fragments[fragid]))
        self.modintcrd = self.modintcrd.tplcorrect(typo, level, indmap)


    def _getdis(self, ith, jth):
        return self.__dmat[ith, jth]

    def _getang(self, ith, jth, kth):
        return(angle(self.__rvec, ith, jth, kth))

    def _getdih(self, ith, jth, kth, lth):
        return(dihedral(self.__rvec, ith, jth, kth, lth))

    def _updateintcrdtype(self):
        # BUG only the bond are setted!
        for key in self.intcrd.vals.keys():
            if key == 'R':
                for _intcrd in self.intcrd.vals[key]:
                    _tps = ''
                    _tmp = [self.atlabs[x] for x in _intcrd.indx]
                    _tmp.sort()
                    _tps = _tps.join(_tmp)
                    _intcrd.tps = _tps

    def computeredundant(self, startred, replace=False):
        """Given definitions of internal coordinates returns their values

        Arguments:
            startred {[type]} -- [description]
        """
        # BUG to check lot of stuff
        res = {}
        functionmap = {'R': self._getdis, 'A': self._getang, 'D': self._getdih}
        for key in startred.keys():
            if key != 'L':
                res[key] = []
                for indices in startred[key]:
                    res[key].append(IntCoordinate(indices.indx,
                                                  functionmap[key](*indices.indx), None))
        if replace:
            self.intcrd = SetIntCoord(res)
        else:
            return res

    def reorderas(self, other, recalcred=False):
        if len(self.atoms) != len(other.atoms):
            print('Different molecules or fragments')
            return None
        if sorted(list(set(self.atoms))) != sorted(list(set(other.atoms))):
            print('Different molecules or fragments')
            return None
        crd_tmp_gss = copy.deepcopy(self.atmcrd) - rmsd.centroid(self.atmcrd)
        crd_tmp_ref = copy.deepcopy(other.atmcrd) - rmsd.centroid(other.atmcrd)
        atm_tmp_ref = np.array(other.atlabs)
        atm_tmp_gss = np.array(self.atlabs)
        result_rmsd, q_swap, q_reflection, q_review = rmsd.check_reflections(atm_tmp_ref, atm_tmp_gss,
                                                                             crd_tmp_ref, crd_tmp_gss,
                                                                             reorder_method=rmsd.reorder_hungarian,
                                                                             rotation_method=rmsd.kabsch_rmsd,
                                                                             keep_stereo=True)
        if result_rmsd > 0.35:
            print('Different molecules or fragments')
            return None
        self.atoms = self.atoms[q_review]
        self.atmcrd = self.atmcrd[q_review]
        if other.intcrd and recalcred:
            self.computeredundant(other.intcrd, replace=True)
        elif self.intcrd:
            indmap = {x:y for y,x in enumerate(q_review)}
            for key in self.__intcrd.vals.keys():
            # FIXME Linear coordinate discharged
                if key != 'L':
                    for valz in self.__intcrd.vals[key]:
                        tmp_indx = copy.deepcopy(valz.indx)
                        valz.indx = [indmap[x] for x in tmp_indx]


def printmod_gic(moldata):
    """
    prints for gaussian mod
    """
    toprint = ''
    modify = moldata.modintcrd.get_mod()
    tmplstrg = '{}mod{}(Value={:7.5f},{}={:7.5f},freeze)={}\n'
    tols = {'R': 0.1,
            'A': 0.2,
            'D': 0.2}
    if modify:
        for key in modify.keys():
            cnt = 0
            for indx in modify[key]:
                cnt += 1
                oldval = moldata.intcrd.vals[key][indx].value
                newval = moldata.modintcrd.vals[key][indx].value
                delta = (newval-oldval) * tols[key]
                indrep = key+'('
                for ind in moldata.modintcrd.vals[key][indx].indx:
                    indrep += '{:d},'.format(ind+1)
                indrep = indrep[:-1]+')'
                if delta > 0:
                    thrkey = 'min'
                else:
                    thrkey = 'max'
                thrval = newval - delta
                toprint += tmplstrg.format(key, cnt, newval,
                                           thrkey, thrval,
                                           indrep)
    return toprint


























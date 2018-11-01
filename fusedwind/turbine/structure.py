
import os
import time
import re
import numpy as np
from scipy.interpolate import pchip

from openmdao.api import Component, Group, ParallelGroup
from openmdao.core.problem import Problem
from openmdao.api import IndepVarComp

from fusedwind.turbine.geometry import FFDSpline
from collections import OrderedDict

try:
    from PGL.components.airfoil import AirfoilShape
    from PGL.main.geom_tools import curvature
    from PGL.main.domain import Domain, Block
    from PGL.main.curve import Curve
    _PGL_installed = True
except:
    print('Warning: PGL not installed, some components will not function correctly')
    _PGL_installed = False


def read_bladestructure(filebase):
    """
    input file reader of BladeStructureVT3D data

    parameters
    ----------
    filebase: str
        data files' basename

    returns
    -------
    st3d: dict
        dictionary containing geometric and material properties
        definition of the blade structure
    """

    def _check_file_version(st3d, headerline):
        ''' Checks the version string of the first line in file

        :param st3d: The dictionary beeing filled
        :param headerline: First line if the file.
        :return: version int, i.e. 1 for a header with '# version 1'
        '''

        if 'version' in [char for char in headerline]:
            # we have a file that is in version numbering
            version = int(headerline[1])
            # check for files consistency
            if version != st3d['version'] and st3d['version'] is not None:
                print('Warning: Files not all consistent in version %s!' % version)

            st3d['version'] = version
        else:
            version = 0
            # check for files consistency
            if version != st3d['version'] and st3d['version'] is not None:
                print('Warning: Files not all consistent in version %s!' % version)

            st3d['version'] = version # version 0 for files before file version tagging
        return version

    def _check_bondline(headerline):
        ''' Checks the version string of the first line in file
        :param headerline: Second line if the file.
        :return: boolean if bondline exists in file set'
        '''
        if 'bond00' in [char for char in headerline]:
            return True
        else:
            return False

    st3d = {}
    st3d['version'] = None
    # read mat file
    fid = open(filebase + '.mat', 'r')
    first_line = fid.readline().split()[1:]
    version = _check_file_version(st3d, first_line)
    if version == 0:
        materials = first_line
    if version >= 1:
        materials = fid.readline().split()[1:]
    #st3d['materials'] = {name:i for i, name in enumerate(materials)}
    st3d['materials'] = OrderedDict()
    for i, name in enumerate(materials):
        st3d['materials'][name] = i
    data = np.loadtxt(fid)
    st3d['matprops'] = data

    # read failmat file
    failcrit = {1:'maximum_strain', 2:'maximum_stress', 3:'tsai_wu'}
    fid = open(filebase + '.failmat', 'r')
    first_line = fid.readline().split()[1:]
    version = _check_file_version(st3d, first_line)
    if version == 0:
        materials = first_line
    if version >= 1:
        materials = fid.readline().split()[1:]
    data = np.loadtxt(fid)
    st3d['failmat'] = data[:, 1:]
    st3d['failcrit'] = [failcrit[mat] for mat in data[:, 0]]
    # read the dp3d file containing region division points
    dpfile = filebase + '.dp3d'

    dpfid = open(dpfile, 'r')
    first_line = dpfid.readline().split()[1:]
    version = _check_file_version(st3d, first_line)

    # read webs and bonds
    if version == 0:
        wnames = first_line
        bondline = False
    if version >= 1:
        second_line = dpfid.readline().split()[1:]
        bondline = _check_bondline(second_line)
        if bondline:
            bnames = second_line
            ibonds = []
            for b, bname in enumerate(bnames):
                line = dpfid.readline().split()[1:]
                line = [int(entry) for entry in line]
                ibonds.append(line)
            st3d['bond_def'] = ibonds
            nbonds = len(ibonds)
            wnames = dpfid.readline().split()[1:]
        else:
            wnames = second_line
    iwebs = []
    for w, wname in enumerate(wnames):
        line = dpfid.readline().split()[1:]
        line = [int(entry) for entry in line]
        iwebs.append(line)
    st3d['web_def'] = iwebs
    nwebs = len(iwebs)
    header = dpfid.readline()
    dpdata = np.loadtxt(dpfile)
    nreg = dpdata.shape[1] - 2
    try:
        regions = header.split()[1:]
        assert len(regions) == nreg
    except:
        regions = ['region%02d' % i for i in range(nreg)]
    st3d['s'] = dpdata[:, 0]
    st3d['DPs'] = dpdata[:, 1:]

    # check if a geo3d file containing region param2 input exists
    pfile = filebase + '.geo3d'
    st3d['dominant_regions'] = []
    st3d['cap_DPs'] = []
    st3d['le_DPs'] = []
    st3d['te_DPs'] = []
    if os.path.exists(pfile):
        pfid = open(pfile, 'r')
        first_line = pfid.readline().split()[1:]
        version = _check_file_version(st3d, first_line)

        st3d['struct_angle'] = float(pfid.readline().split()[2])
        line = pfid.readline().split()[2:]
        st3d['cap_DPs'] = [int(entry) for entry in line]
        line = pfid.readline().split()[2:]
        st3d['te_DPs'] = [int(entry) for entry in line]
        line = pfid.readline().split()[2:]
        st3d['le_DPs'] = [int(entry) for entry in line]
        line = pfid.readline().split()[2:]
        st3d['dominant_regions'] = [int(entry) for entry in line]
        header = pfid.readline().split()[1:]
        # ensure that header contains required names
        assert len(header) >= 7
        nweb = len(header) - 7
        data = np.loadtxt(pfid)
        assert np.allclose(st3d['s'], data[:, 0], rtol=1.e-5)
        for i, name in enumerate(header[1:]):
            st3d[name] = data[:, i+1]
        pfid.close()

    # read the st3d files containing thicknesses and orientations
    st3d['regions'] = []
    for i, rname in enumerate(regions):
        r = {}
        layup_file = '_'.join([filebase, rname]) + '.st3d'
        fid = open(layup_file, 'r')
        first_line = fid.readline().split()[1:]
        version = _check_file_version(st3d, first_line)
        if version == 0:
            rrname = first_line
        if version >= 1:
            rrname = fid.readline().split()[1]
        lheader = fid.readline().split()[1:]

        cldata = np.loadtxt(fid)
        layers = lheader[1:]
        nl = len(layers)

        if version==0:
            # check that layer names are of the type <%s><%02d>
            lnames = []
            basenames = []
            for name in layers:
                try:
                    # numbers in names should be allowed
                    split = re.match(r"([a-z]+)([0-9]+)", name, re.I).groups()
                    idx = basenames.count(split[0])
                    basenames.append(split[0])
                    lnames.append(split[0] + '%02d' % idx)
                except:
                    split = re.match(r"([a-z]+)", name, re.I).groups()
                    idx = basenames.count(split[0])
                    basenames.append(split[0])
                    lnames.append(split[0] + '%02d' % idx)
            r['layers'] = lnames

        if version >= 1:
            r['layers'] = layers

        r['thicknesses'] = cldata[:, 1:nl + 1]
        if cldata.shape[1] == nl*2 + 1:
            r['angles'] = cldata[:, nl + 1:2*nl+1 + 2]
        else:
            r['angles'] = np.zeros((cldata.shape[0], nl))
        st3d['regions'].append(r)

    st3d['webs'] = []
    for i, rname in enumerate(wnames):
        r = {}
        layup_file = '_'.join([filebase, rname]) + '.st3d'
        fid = open(layup_file, 'r')
        first_line = fid.readline().split()[1:]
        version = _check_file_version(st3d, first_line)
        if version == 0:
            rrname = first_line
        if version >= 1:
            rrname = fid.readline().split()[1]
        lheader = fid.readline().split()[1:]

        cldata = np.loadtxt(fid)
        layers = lheader[1:]
        nl = len(layers)

        if version == 0:
            # check that layer names are of the type <%s><%02d>
            lnames = []
            basenames = []
            for name in layers:
                try:
                    # numbers in names should be allowed
                    split = re.match(r"([a-z]+)([0-9]+)", name, re.I).groups()
                    idx = basenames.count(split[0])
                    basenames.append(split[0])
                    lnames.append(split[0] + '%02d' % idx)
                except:
                    split = re.match(r"([a-z]+)", name, re.I).groups()
                    idx = basenames.count(split[0])
                    basenames.append(split[0])
                    lnames.append(split[0] + '%02d' % idx)
            r['layers'] = lnames

        if version >= 1:
            r['layers'] = layers

        r['thicknesses'] = cldata[:, 1:nl + 1]
        if cldata.shape[1] == nl*2 + 1:
            r['angles'] = cldata[:, nl + 1:2*nl+1 + 2]
        else:
            r['angles'] = np.zeros((cldata.shape[0], nl))
        st3d['webs'].append(r)

    if bondline:
        st3d['bonds'] = []
        for i, rname in enumerate(bnames):
            r = {}
            layup_file = '_'.join([filebase, rname]) + '.st3d'
            fid = open(layup_file, 'r')
            first_line = fid.readline().split()[1:]
            #version = _check_file_version(st3d, first_line)
            rrname = fid.readline().split()[1]
            lheader = fid.readline().split()[1:]

            cldata = np.loadtxt(fid)
            layers = lheader[1:]
            nl = len(layers)

            r['layers'] = layers

            r['thicknesses'] = cldata[:, 1:nl + 1]
            if cldata.shape[1] == nl*2 + 1:
                r['angles'] = cldata[:, nl + 1:2*nl+1 + 2]
            else:
                r['angles'] = np.zeros((cldata.shape[0], nl))
            st3d['bonds'].append(r)

    return st3d


def write_bladestructure(st3d, filebase):
    """
    input file writer for a blade structure definition

    parameters
    ----------
    st3d: dict
        dictionary containing geometric and material properties
        definition of the blade structure
    filebase: str
        data files' basename
    """

    # write material properties
    fid = open(filebase + '.mat', 'w')
    fid.write('# version %s\n' % st3d['version'])
    fid.write('# %s\n' % (' '.join(st3d['materials'].keys())))
    fid.write('# E1 E2 E3 nu12 nu13 nu23 G12 G13 G23 rho\n')
    fmt = ' '.join(10*['%.20e'])
    np.savetxt(fid, st3d['matprops'], fmt=fmt)

    failcrit = dict(maximum_strain=1, maximum_stress=2, tsai_wu=3)
    fid = open(filebase + '.failmat', 'w')
    fid.write('# version %s\n' % st3d['version'])
    fid.write('# %s\n' % (' '.join(st3d['materials'])))
    fid.write('# failcrit s11_t s22_t s33_t s11_c s22_c s33_c'
              't12 t13 t23 e11_t e22_t e33_t e11_c e22_c e33_c g12 g13 g23'
              'gM0 C1a C2a C3a C4a\n')
    data = np.zeros((st3d['failmat'].shape[0], st3d['failmat'].shape[1]+1))
    data[:, 0] = [failcrit[mat] for mat in st3d['failcrit']]
    data[:, 1:] = st3d['failmat']
    fmt = '%i ' + ' '.join(23*['%.20e'])
    np.savetxt(fid, np.asarray(data), fmt=fmt)

    # write dp3d file with region division points
    fid = open(filebase + '.dp3d', 'w')
    fid.write('# version %s\n' % st3d['version'])
    if 'bonds' in st3d:
        bonds = ['bond%02d' % i for i in range(len(st3d['bonds']))]
        fid.write('# %s\n' % ('  '.join(bonds)))
        for bond in st3d['bond_def']:
            fid.write('# %i %i %i %i\n' % (bond[0], bond[1], bond[2], bond[3]))
    webs = ['web%02d' % i for i in range(len(st3d['webs']))]
    fid.write('# %s\n' % ('  '.join(webs)))
    for web in st3d['web_def']:
        fid.write('# %i %i\n' % (web[0], web[1]))
    DPs = ['DP%02d' % i for i in range(st3d['DPs'].shape[1])]
    fid.write('# s %s\n' % (' '.join(DPs)))
    data = np.array([st3d['s']]).T
    data = np.append(data, st3d['DPs'], axis=1)
    np.savetxt(fid, data)
    fid.close()

    # write geo3d file
    if 'cap_width_ps' in st3d.keys():
        fid = open(filebase + '.geo3d', 'w')
        fid.write('# version %s\n' % st3d['version'])
        fid.write('# struct_angle %24.15f\n' % st3d['struct_angle'])
        fid.write('# cap_DPs %s\n' % ('  '.join(map(str, st3d['cap_DPs']))))
        fid.write('# te_DPs %s\n' % ('  '.join(map(str, st3d['te_DPs']))))
        fid.write('# le_DPs %s\n' % ('  '.join(map(str, st3d['le_DPs']))))
        fid.write('# dominant_regions %s\n' % ('  '.join(map(str, st3d['dominant_regions']))))
        header = ['s', 'cap_center_ps',
                       'cap_center_ss',
                       'cap_width_ps',
                       'cap_width_ss',
                       'te_width',
                       'le_width']
        header.extend(['w%02dpos' % i for i in range(1, len(st3d['web_def']))])
        fid.write('# %s\n' % (' '.join(header)))
        data = np.array([st3d[name] for name in header]).T
        np.savetxt(fid, data)
        fid.close()

    # write st3d files with material thicknesses and angles
    for i, reg in enumerate(st3d['regions']):
        rname = 'region%02d' % i
        fname = '_'.join([filebase, rname]) + '.st3d'
        fid = open(fname, 'w')
        fid.write('# version %s\n' % st3d['version'])
        lnames = '    '.join(reg['layers'])
        fid.write('# %s\n' % rname)
        fid.write('# s    %s\n' % lnames)
        data = np.array([st3d['s']]).T
        data = np.append(data, reg['thicknesses'], axis=1)
        data = np.append(data, reg['angles'], axis=1)
        np.savetxt(fid, data)
        fid.close()
    for i, reg in enumerate(st3d['webs']):
        rname = 'web%02d' % i
        fname = '_'.join([filebase, rname]) + '.st3d'
        fid = open(fname, 'w')
        fid.write('# version %s\n' % st3d['version'])
        lnames = '    '.join(reg['layers'])
        fid.write('# %s\n' % rname)
        fid.write('# s    %s\n' % lnames)
        data = np.array([st3d['s']]).T
        data = np.append(data, reg['thicknesses'], axis=1)
        data = np.append(data, reg['angles'], axis=1)
        np.savetxt(fid, data)
        fid.close()
    if 'bonds' in st3d:
        for i, reg in enumerate(st3d['bonds']):
            rname = 'bond%02d' % i
            fname = '_'.join([filebase, rname]) + '.st3d'
            fid = open(fname, 'w')
            fid.write('# version %s\n' % st3d['version'])
            lnames = '    '.join(reg['layers'])
            fid.write('# %s\n' % rname)
            fid.write('# s    %s\n' % lnames)
            data = np.array([st3d['s']]).T
            data = np.append(data, reg['thicknesses'], axis=1)
            data = np.append(data, reg['angles'], axis=1)
            np.savetxt(fid, data)
            fid.close()


def interpolate_bladestructure(st3d, s_new):
    """
    interpolate a blade structure definition onto
    a new spanwise distribution using pchip

    parameters
    ----------
    st3d: dict
        dictionary with blade structural definition
    s_new: array
        1-d array with new spanwise distribution

    returns
    -------
    st3dn: dict
        blade structural definition interpolated onto s_new distribution
    """

    st3dn = {}
    sorg = st3d['s']
    st3dn['s'] = s_new.copy()
    st3dn['version'] = st3d['version']
    st3dn['materials'] = st3d['materials']
    st3dn['matprops'] = st3d['matprops']
    st3dn['failmat'] = st3d['failmat']
    st3dn['failcrit'] = st3d['failcrit']
    st3dn['web_def'] = st3d['web_def']
    st3dn['dominant_regions'] = st3d['dominant_regions']
    st3dn['cap_DPs'] = st3d['cap_DPs']
    st3dn['le_DPs'] = st3d['le_DPs']
    st3dn['te_DPs'] = st3d['le_DPs']
    try:
        st3dn['struct_angle'] = st3d['struct_angle']
        st3dn['cap_DPs'] = st3d['cap_DPs']
        st3dn['te_DPs'] = st3d['te_DPs']
        st3dn['le_DPs'] = st3d['le_DPs']
        names = ['cap_center_ps',
                 'cap_center_ss',
                 'cap_width_ps',
                 'cap_width_ss',
                 'te_width',
                 'le_width']
        names.extend(['w%02dpos' % i for i in range(1, len(st3d['web_def']))])
        for name in names:
            tck = pchip(sorg, st3d[name])
            st3dn[name] = tck(s_new)
    except:
        print 'no geo3d data'

    st3dn['regions'] = []
    st3dn['webs'] = []
    if 'bonds' in st3d:
        st3dn['bonds'] = []
        st3dn['bond_def'] = st3d['bond_def']

    DPs = np.zeros((s_new.shape[0], st3d['DPs'].shape[1]))
    for i in range(st3d['DPs'].shape[1]):
        tck = pchip(sorg, st3d['DPs'][:, i])
        DPs[:, i] = tck(s_new)
    st3dn['DPs'] = DPs

    for r in st3d['regions']:
        rnew = {}
        rnew['layers'] = r['layers']
        Ts = r['thicknesses']
        As = r['angles']
        tnew = np.zeros((s_new.shape[0], Ts.shape[1]))
        anew = np.zeros((s_new.shape[0], As.shape[1]))
        for i in range(Ts.shape[1]):
            tck = pchip(sorg, Ts[:, i])
            tnew[:, i] = tck(s_new)
            tck = pchip(sorg, As[:, i])
            anew[:, i] = tck(s_new)
        rnew['thicknesses'] = tnew.copy()
        rnew['angles'] = anew.copy()
        st3dn['regions'].append(rnew)
    for r in st3d['webs']:
        rnew = {}
        rnew['layers'] = r['layers']
        Ts = r['thicknesses']
        As = r['angles']
        tnew = np.zeros((s_new.shape[0], Ts.shape[1]))
        anew = np.zeros((s_new.shape[0], As.shape[1]))
        for i in range(Ts.shape[1]):
            tck = pchip(sorg, Ts[:, i])
            tnew[:, i] = tck(s_new)
            tck = pchip(sorg, As[:, i])
            anew[:, i] = tck(s_new)
        rnew['thicknesses'] = tnew.copy()
        rnew['angles'] = anew.copy()
        st3dn['webs'].append(rnew)
    if 'bonds' in st3d:
        for r in st3d['bonds']:
            rnew = {}
            rnew['layers'] = r['layers']
            Ts = r['thicknesses']
            As = r['angles']
            tnew = np.zeros((s_new.shape[0], Ts.shape[1]))
            anew = np.zeros((s_new.shape[0], As.shape[1]))
            for i in range(Ts.shape[1]):
                tck = pchip(sorg, Ts[:, i])
                tnew[:, i] = tck(s_new)
                tck = pchip(sorg, As[:, i])
                anew[:, i] = tck(s_new)
            rnew['thicknesses'] = tnew.copy()
            rnew['angles'] = anew.copy()
            st3dn['bonds'].append(rnew)

    return st3dn


class ComputeDPsParam2(object):
    """
    Computes the region DPs based on the lofted surface and below
    listed parameters

    parameters
    ----------
    surface: array
        lofted blade surface
    le_width: array
        total width of leading edge panel across leading edge
        as function of span
    te_width: array
        width of trailing edge panels on lower and upper surfaces
        as function of span
    cap_center_ss: array
        upper surface cap center position relative to reference axis
        as function of span
    cap_center_ps: array
        lower surface cap center position relative to reference axis
        as function of span
    cap_width_ss: array
        width of upper spar cap as function of span
    cap_width_ps: array
        width of lower spar cap as function of span
    w<%02d>pos: array
        Distance between web and cap center.
        Web attachment DPs are given in the st3d['web_def']
        dict passed during init.
    te_DPs: list
        list of DP indices identifying the trailing edge
        reinforcement position. list should contain
        [lower_surface_DP, upper_surface_DP]
    le_DPs: list
        list of DP indices identifying the leading edge
        reinforcement position. list should contain
        [lower_surface_DP, upper_surface_DP]
    cap_DPs: list
        list of DPs identifying the edges of the spar caps.
        list should be numbered with increasing DP indices,
        i.e. starting at lower surface rear,
        ending at upper surface rear.

    returns
    -------
    DPs: array
        Division points as function of span, size ((10, ni))
    """

    def __init__(self, st3d=None, **kwargs):
        """
        Parameters
        ----------
        st3d: dict
            blade structure definition (optional)
        kwargs: args
            init arguments to optionally set the class variables
        """

        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.surface = np.array([])
        try:
            self.web_def = st3d['web_def']
            self.DPs = st3d['DPs']
            self.s = st3d['s']
            self.te_DPs = st3d['te_DPs']
            self.le_DPs = st3d['le_DPs']
            self.cap_DPs = st3d['cap_DPs']
            self.dominant_regions = st3d['dominant_regions']
            self.le_width = st3d['le_width']
            self.te_width = st3d['te_width']
            self.cap_width_ss = st3d['cap_width_ss']
            self.cap_width_ps = st3d['cap_width_ps']
            self.cap_center_ss = st3d['cap_center_ss']
            self.cap_center_ps = st3d['cap_center_ps']
            self.struct_angle = st3d['struct_angle']
            for i, web_ix in enumerate(self.web_def[1:]):
                name = 'w%02dpos' % (i+1)
                setattr(self, name, st3d[name])
        except:
            print 'failed reading st3d'
            self.web_def = []
            self.DPs = np.array([])
            self.s = np.array([])
            self.te_DPs = []
            self.le_DPs = []
            self.cap_DPs = []
            self.le_width = np.array([])
            self.te_width = np.array([])
            self.cap_width_ss = np.array([])
            self.cap_width_ps = np.array([])
            self.cap_center_ss = np.array([])
            self.cap_center_ps = np.array([])
            self.struct_angle = 0.
            self.dominant_regions = []
            for i, web_ix in enumerate(self.web_def):
                setattr(self, 'w%02dpos' % i, np.array([]))

        self.consistency_check = True
        self.min_width = 0.
        self.ref_axis = 'linear'

        for k, v in kwargs.iteritems():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print 'unknown key %s' % k

    def compute(self):

        self.ni = self.surface.shape[1]

        self.dom = Domain()
        self.dom.add_blocks(Block(self.surface[:, :, 0],
                                  self.surface[:, :, 1],
                                  self.surface[:, :, 2]))
        surforg = self.dom.blocks['block-00']._block2arr()[:, :, 0, :].copy()
        self.dom.rotate_z(self.struct_angle)
        if self.ref_axis == 'main_axis':
            x = self.x
            y = self.y
        elif self.ref_axis == 'linear':
            x = np.interp(self.z, np.linspace(0, 1, self.ni),
                                  np.linspace(self.x[0], self.x[-1], self.ni))
            y = np.interp(self.z, np.linspace(0, 1, self.ni),
                                  np.linspace(self.y[0], self.y[-1], self.ni))
        else:
            raise RuntimeError('ref_axis %s not understood' % self.ref_axis)

        self.pitch_axis = Curve(points=np.array([x, y, self.z]).T)
        self.pitch_axis.rotate_z(self.struct_angle)

        surf = self.dom.blocks['block-00']._block2arr()[:, :, 0, :].copy()

        self.afsorg = []
        for i in range(surforg.shape[1]):
            self.afsorg.append(AirfoilShape(points=surforg[:, i, :]))

        self.afs = []
        for i in range(surf.shape[1]):
            self.afs.append(AirfoilShape(points=surf[:, i, :]))

        if len(self.cap_DPs) == 0:
            raise RuntimeError('cap_DPs not specified')
        if len(self.le_DPs) == 0:
            raise RuntimeError('le_DPs not specified')
        if len(self.te_DPs) == 0:
            raise RuntimeError('te_DPs not specified')

        DPs = self.DPs

        # trailing edge DPs
        DPs[:, 0] = -1.
        DPs[:, -1] = 1.

        # set extra TE regions if user hasn't specified anything
        if (DPs[:, 1] == 0.).all():
            DPs[:, 1] = -0.99
            DPs[:, -2] = 0.99

        for i in range(self.ni):
            af = self.afs[i]

            # Leading edge
            sLE = af.sLE
            lew = self.le_width[i]
            w0 = lew / 2. / af.smax
            s0 = sLE - w0
            s1 = sLE + w0
            s0 = -1.0 + s0 / sLE
            s1 = (s1-sLE) / (1.0-sLE)
            DPs[i, self.le_DPs[0]] = s0
            DPs[i, self.le_DPs[1]] = s1

            # lower trailing panel
            TEp = self.te_width[i]
            DPs[i, self.te_DPs[0]] = -1. + TEp / (sLE * af.smax)
            # upper trailing panel
            DPs[i, self.te_DPs[1]] = 1. - TEp / ((1. - sLE) * af.smax)

        # cap DPs
        PAx = self.pitch_axis.points[:, 0]
        for i in range(self.ni):
            af = self.afs[i]
            x_ccL = PAx[i] + self.cap_center_ps[i]
            s_ccL = af.interp_x(x_ccL, 'lower')
            x_ccU = PAx[i] + self.cap_center_ss[i]
            s_ccU = af.interp_x(x_ccU, 'upper')
            cwU = self.cap_width_ss[i] / 2 / af.smax
            cwL = self.cap_width_ps[i] / 2 / af.smax
            DPs[i, self.cap_DPs[0]] = af.s_to_11(s_ccL - cwL)
            DPs[i, self.cap_DPs[1]] = af.s_to_11(s_ccL + cwL)
            DPs[i, self.cap_DPs[2]] = af.s_to_11(s_ccU - cwU)
            DPs[i, self.cap_DPs[3]] = af.s_to_11(s_ccU + cwU)

            # web DPs
            x_cc = PAx[i]
            s_ccL = af.interp_x(x_cc, 'lower')
            s_ccU = af.interp_x(x_cc, 'upper')
            for j, web_ix in enumerate(self.web_def[1:]):
                wacc = getattr(self, 'w%02dpos' % (j+1))[i]
                swL = af.interp_x(x_cc+wacc, 'lower')
                swU = af.interp_x(x_cc+wacc, 'upper')
                DPs[i, web_ix[0]] = af.s_to_11(swL)
                DPs[i, web_ix[1]] = af.s_to_11(swU)

        if self.consistency_check == True:

            self.check_consistency()

    def check_consistency(self):
        """
        check that there are no negative region widths
        and that DPs belonging to ps and ss are not
        on the wrong side of the LE.

        If negative widths are identified either of two things will be done:
        | 1) for all regular DPs the midpoint between the DPs is
        identified and the DPs are placed +/- 0.01% curve length
        to either side of this point.
        | 2) if one of the DPs is a dominant DP, the neighbour DP will be shifted
        by 0.02% curve length.
        """

        DPs = self.DPs
        # check that ps and ss DPs are on the correct sides of the LE
        ps = range(self.te_DPs[0], self.le_DPs[0] + 1)
        ss = range(self.le_DPs[1], self.te_DPs[1] + 1)
        for i in range(self.ni):
            DPs[i, :] = np.maximum(-1., DPs[i,:])
            DPs[i, :] = np.minimum(1., DPs[i,:])
            for j in ps:
                try:
                    min_width = self.min_width[j]
                except:
                    min_width = self.min_width
                if self.afs[i].s_to_01(DPs[i, j]) > self.afs[i].sLE:
                    DPs[i, j] = self.afs[i].s_to_11(self.afs[i].sLE) - min_width
            for j in ss:
                try:
                    min_width = self.min_width[j]
                except:
                    min_width = self.min_width
                if self.afs[i].s_to_01(DPs[i, j]) < self.afs[i].sLE:
                    DPs[i, j] = self.afs[i].s_to_11(self.afs[i].sLE)  + min_width

        # check for negative region widths
        ps = range(1, self.le_DPs[0] + 1)
        ss = range(self.le_DPs[1], self.te_DPs[1]+2)
        psss = [ps, ss[::-1]]
        self.DPs_save = DPs.copy()

        for i in range(self.ni):
            for side in [0, 1]:
                sgn = -1.*side + (1-side)*1.
                for jj,j in enumerate(psss[side]):
                    try:
                        min_width = self.min_width[j]
                    except:
                        min_width = self.min_width
                    for k in psss[side][jj+1:]:
                        if sgn * np.diff(DPs[i, [j, k]]) < 0.:
                            if j-1*(1-side) in self.dominant_regions and k not in self.cap_DPs:
                                DPs[i, k] = DPs[i, j] + min_width
                            elif j-1*(1-side) in self.dominant_regions and k in self.cap_DPs:
                                DPs[i, j] = DPs[i, k] + min_width
                            elif k-1*side in self.dominant_regions and j not in self.cap_DPs:
                                DPs[i, j] = DPs[i, k] - min_width
                            else:
                                mid = 0.5 * (DPs[i, j] + DPs[i, k])
                                DPs[i, j] = mid - min_width
                                DPs[i, k] = mid + min_width

    def plot(self, isec=None, ifig=1, coordsys='rotor'):

        import matplotlib.pylab as plt

        if coordsys == 'rotor':
            afs = self.afsorg
        elif coordsys == 'mold':
            afs = self.afs

        plt.figure(ifig)

        if isec is not None:
            ni = [isec]
        else:
            ni = range(self.ni)

        for i in ni:
            plt.title('r = %3.3f' % (self.z[i]))
            af = afs[i]
            plt.plot(af.points[:, 0], af.points[:, 1], 'b-')
            DP = np.array([af.interp_s(af.s_to_01(s)) for s in self.DPs[i, :]])
            width = np.diff(self.DPs[i, :])
            valid = np.ones(DP.shape[0])
            try:
                valid[1:] = width > self.min_width[1:]
            except:
                valid[1:] = width > self.min_width
            for d in DP:
                plt.plot(d[0], d[1], 'ro')
            for d in DP[self.cap_DPs, :]:
                plt.plot(d[0], d[1], 'mo')
            for web_ix in self.web_def:
                if valid[web_ix].all():
                    plt.plot(DP[[web_ix[0], web_ix[1]]][:, 0],
                             DP[[web_ix[0], web_ix[1]]][:, 1], 'g')

        plt.axis('equal')

    def plot_topview(self, coordsys='rotor', ifig=None):

        import matplotlib.pylab as plt

        if coordsys == 'rotor':
            afs = self.afsorg
        elif coordsys == 'mold':
            afs = self.afs

        plt.figure(ifig)
        DPs = []
        for i in range(self.ni):
            af = afs[i]
            plt.plot(af.points[:, 2], af.points[:, 0])
            DP = [af.interp_s(af.s_to_01(s)) for s in self.DPs[i, :]]
            DPs.append(DP)
            for d in DP:
                plt.plot(d[2], d[0], 'ro')

        self.DPs_xyz = np.asarray(DPs)
        #plt.show()


class SplinedBladeStructureBase(Group):
    """
    class that adds structural geometry variables to the analysis
    either as splines with user defined control points
    or arrays according to the initial structural data
    """

    def __init__(self, st3d):
        """
        parameters
        ----------
        st3d: dict
            dictionary with blade structural definition
        """
        super(SplinedBladeStructureBase, self).__init__()

        self._vars = []
        self._allvars = []
        self.st3dinit = st3d

        # add materials properties array ((10, nmat))
        self.add('matprops_c', IndepVarComp('matprops', st3d['matprops']), promotes=['*'])

        # add materials strength properties array ((18, nmat))
        self.add('failmat_c', IndepVarComp('failmat', st3d['failmat']), promotes=['*'])

    def _add_mat_spline(self, names, Cx, spline_type='bezier', scaler=1.):
        """
        adds a 1D FFDSpline for the given variable
        with user defined spline type and control point locations.

        parameters
        ----------
        name: str or tuple
            name of the variable(s), which should be of the form
            `r04uniaxT` or `r04uniaxA` for region 4 uniax thickness
            and angle, respectively. if `name` is a list of names,
            spline CPs will be grouped.
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip

        examples
        --------
        | name: DP04 results in spline CPs indepvar: DP04_C,
        | name: r04uniaxT results in spline CPs indepvar: r04uniaxT_C,
        | name: (r04uniaxT, r04uniax01T) results in spline CPs: r04uniaxT_C
        which controls both thicknesses as a group.
        """

        st3d = self.st3dinit
        tvars = []

        for name in names:
            if name.startswith('r') or name.startswith('w') or name.startswith('b'):
                l_index = None
                try:
                    ireg = int(name[1:3])
                    try:
                        split = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name[3:], re.I).groups()
                        l_index = split[1]
                    except:
                        split = re.match(r"([a-z]+)([a-z]+)", name[3:], re.I).groups()
                    layername = split[0]+split[1]
                    stype = split[-1]
                except:
                    raise RuntimeError('Variable name %s not understood' % name)
            else:
                raise RuntimeError('Variable name %s not understood' % name)

            if name.startswith('r'):
                r = st3d['regions'][ireg]
                rname = 'r%02d' % ireg
            elif name.startswith('w'):
                r = st3d['webs'][ireg]
                rname = 'w%02d' % ireg
            elif name.startswith('b'):
                r = st3d['bonds'][ireg]
                rname = 'b%02d' % ireg

            varname = '%s%s%s' % (rname, layername, stype)
            ilayer = r['layers'].index(layername)
            if stype == 'T':
                var = r['thicknesses'][:, ilayer]
            elif stype == 'A':
                var = r['angles'][:, ilayer]
            c = self.add(name + '_s', FFDSpline(name, st3d['s'],
                                                   var,
                                                   Cx, scaler=scaler),
                                                   promotes=[name])
            c.spline_options['spline_type'] = spline_type
            c.set_spline(spline_type)
            tvars.append(name)

        self._vars.extend(tvars)
        # finally add the IndepVarComp and make the connections
        self.add(names[0] + '_c', IndepVarComp(names[0] + '_C', np.zeros(len(Cx))), promotes=['*'])
        for varname in tvars:
            self.connect(names[0] + '_C', varname + '_s.' + varname + '_C')

    def configure(self):

        print 'SplinedBladeStructure: No harm done, but configure is depreciated\n' + \
              'and replaced by pre_setup called automatically by OpenMDAO.\n'+\
              'Ensure that you have OpenMDAO > v1.7.1 installed'

    def pre_setup(self, problem):
        """
        add IndepVarComp's for all remaining planform variables
        """

        st3d = self.st3dinit

        for i in range(st3d['DPs'].shape[1]):
            varname = 'DP%02d' % i
            var = st3d['DPs'][:, i]
            if varname not in self._vars:
                self.add(varname + '_c', IndepVarComp(varname, var), promotes=['*'])

        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                if varname+'T' not in self._vars:
                    self.add(varname + 'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                if varname+'A' not in self._vars:
                    self.add(varname + 'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])
        for ireg, reg in enumerate(st3d['webs']):
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                if varname+'T' not in self._vars:
                    self.add(varname + 'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                if varname+'A' not in self._vars:
                    self.add(varname + 'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])
        if 'bonds' in st3d:
            for ireg, reg in enumerate(st3d['bonds']):
                for i, lname in enumerate(reg['layers']):
                    varname = 'b%02d%s' % (ireg, lname)
                    if varname+'T' not in self._vars:
                        self.add(varname + 'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                    if varname+'A' not in self._vars:
                        self.add(varname + 'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])


class SplinedBladeStructure(SplinedBladeStructureBase):
    """
    class that adds structural geometry variables to the analysis
    either as splines with user defined control points
    or arrays according to the initial structural data
    """

    def __init__(self, st3d):
        """
        parameters
        ----------
        st3d: dict
            dictionary with blade structural definition
        """
        super(SplinedBladeStructure, self).__init__(st3d)

    def add_spline(self, name, Cx, spline_type='bezier', scaler=1.):
        """
        adds a 1D FFDSpline for the given variable
        with user defined spline type and control point locations.

        parameters
        ----------
        name: str or tuple
            name of the variable(s), which should be of the form
            `r04uniaxT` or `r04uniaxA` for region 4 uniax thickness
            and angle, respectively. if `name` is a list of names,
            spline CPs will be grouped.
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip

        examples
        --------
        | name: DP04 results in spline CPs indepvar: DP04_C,
        | name: r04uniaxT results in spline CPs indepvar: r04uniaxT_C,
        | name: (r04uniaxT, r04uniax01T) results in spline CPs: r04uniaxT_C
        which controls both thicknesses as a group.
        """

        if isinstance(name, str):
            names = [name]
        else:
            names = name
        # decode the name
        if 'DP' in names[0]:
            self._add_DP_spline(names, Cx, spline_type, scaler=1.)
        else:
            self._add_mat_spline(names, Cx, spline_type, scaler=1.)

    def _add_DP_spline(self, names, Cx, spline_type='bezier', scaler=1.):
        """
        adds a 1D FFDSpline for the given variable
        with user defined spline type and control point locations.

        parameters
        ----------
        name: str or tuple
            name of the variable(s), which should be of the form
            `r04uniaxT` or `r04uniaxA` for region 4 uniax thickness
            and angle, respectively. if `name` is a list of names,
            spline CPs will be grouped.
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip

        examples
        --------
        | name: DP04 results in spline CPs indepvar: DP04_C,
        | name: r04uniaxT results in spline CPs indepvar: r04uniaxT_C,
        | name: (r04uniaxT, r04uniax01T) results in spline CPs: r04uniaxT_C
        which controls both thicknesses as a group.
        """

        st3d = self.st3dinit

        tvars = []
        for name in names:
            try:
                iDP = int(re.match(r"([a-z]+)([0-9]+)", name, re.I).groups()[-1])
            except:
                raise RuntimeError('Variable name %s not understood' % name)

            var = st3d['DPs'][:, iDP]
            c = self.add(name + '_s', FFDSpline(name, st3d['s'],
                                                      var,
                                                      Cx, scaler=scaler),
                                                      promotes=[name])
            c.spline_options['spline_type'] = spline_type
            c.set_spline(spline_type)
            tvars.append(name)
        self._vars.extend(tvars)

        # add the IndepVarComp
        self.add(names[0] + '_c', IndepVarComp(names[0] + '_C', np.zeros(len(Cx))), promotes=['*'])
        for varname in tvars:
            self.connect(names[0] + '_C', varname + '_s.' + varname + '_C')


class DPsParam2(Component):

    def __init__(self, st3d, sdim, config={}):
        super(DPsParam2, self).__init__()

        self._DPs = ComputeDPsParam2(st3d, **config)

        size = sdim[1]

        self.add_param('blade_surface_st', np.zeros(sdim))
        self.add_param('x_st', np.zeros(size))
        self.add_param('y_st', np.zeros(size))
        self.add_param('z_st', np.zeros(size))
        self.add_param('blade_scale', 1.)
        self.add_param('struct_angle', st3d['struct_angle'])
        self._param2_names = ['cap_width_ss',
                              'cap_width_ps',
                              'te_width',
                              'le_width',
                              'cap_center_ss',
                              'cap_center_ps']

        for i in range(len(st3d['web_def'][1:])):
            self._param2_names.append('w%02dpos' % (i + 1))

        for name in self._param2_names:
            self.add_param(name, np.zeros(size))

        self.nDP = st3d['DPs'].shape[1]
        self._vars = []
        for i in range(self.nDP):
            name = 'DP%02d' % i
            self.add_output(name, np.zeros(size))
            self._vars.append(name)

    def solve_nonlinear(self, params, unknowns, resids):

        self._DPs.x = params['x_st']
        self._DPs.y = params['y_st']
        self._DPs.z = params['z_st']
        self._DPs.surface = params['blade_surface_st']
        for name in self._param2_names:
            setattr(self._DPs, name, params[name] / params['blade_scale'])
        self._DPs.struct_angle = params['struct_angle']
        self._DPs.compute()

        for i in range(self.nDP):
            unknowns['DP%02d' % i] = self._DPs.DPs[:, i]


class SplinedBladeStructureParam2(SplinedBladeStructureBase):
    """
    class that adds structural geometry variables to the analysis
    either as splines with user defined control points
    or arrays according to the initial structural data
    """

    def __init__(self, st3d, sdim, config={}):
        """
        parameters
        ----------
        st3d: dict
            dictionary with blade structural definition
        """
        super(SplinedBladeStructureParam2, self).__init__(st3d)

        nDP = st3d['DPs'].shape[1]
        self.nsec = st3d['s'].shape[0]
        promotes = ['blade_surface_st',
                    'x_st', 'y_st', 'z_st',
                    'struct_angle', 'blade_scale']
        for i in range(nDP):
            name = 'DP%02d' % i
            promotes.append(name)
        self.add('compute_dps', DPsParam2(st3d, sdim, config), promotes=promotes)

        self._param2_names = self.compute_dps._param2_names
        # DPsParam2 always outputs DP00, DP01, so add them to
        self._vars = self.compute_dps._vars

    def add_spline(self, name, Cx, spline_type='bezier', scaler=1.):
        """
        adds a 1D FFDSpline for the given variable
        with user defined spline type and control point locations.

        parameters
        ----------
        name: str or tuple
            name of the variable(s), which should be of the form
            `r04uniaxT` or `r04uniaxA` for region 4 uniax thickness
            and angle, respectively. if `name` is a list of names,
            spline CPs will be grouped.
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip

        examples
        --------
        | name: DP04 results in spline CPs indepvar: DP04_C,
        | name: r04uniaxT results in spline CPs indepvar: r04uniaxT_C,
        | name: (r04uniaxT, r04uniax01T) results in spline CPs: r04uniaxT_C
        which controls both thicknesses as a group.
        """

        if isinstance(name, str):
            names = [name]
        else:
            names = name
        # decode the name
        if names[0] in self._param2_names:
            self._add_param2_spline(names, Cx, spline_type=spline_type, scaler=scaler)
        else:
            self._add_mat_spline(names, Cx, spline_type=spline_type, scaler=scaler)

    def _add_param2_spline(self, names, Cx, spline_type='bezier', scaler=1.):

        st3d = self.st3dinit
        tvars = []
        for name in names:
            var = st3d[name]
            c = self.add(name + '_s', FFDSpline(name, st3d['s'],
                                                      var,
                                                      Cx, scaler=scaler),
                                                      promotes=[name])
            c.spline_options['spline_type'] = spline_type
            c.set_spline(spline_type)
            tvars.append(name)
        self._vars.extend(tvars)

        # add the IndepVarComp
        self.add(names[0] + '_c', IndepVarComp(names[0] + '_C', np.zeros(len(Cx))), promotes=['*'])
        for varname in tvars:
            self.connect(names[0] + '_C', varname + '_s.' + varname + '_C')
            self.connect(varname, 'compute_dps.%s' % varname)

    def pre_setup(self, problem):

        self.add('struct_angle_c', IndepVarComp('struct_angle',
                                                self.st3dinit['struct_angle']),
                                                promotes=['*'])

        for varname in self._param2_names:
            if varname not in self._vars:
                var = self.st3dinit[varname]
                self.add(varname + '_c', IndepVarComp(varname, var), promotes=['*'])
                self.connect(varname, 'compute_dps.%s' % varname)
        super(SplinedBladeStructureParam2, self).pre_setup(problem)


class BladeStructureProperties(Component):
    """
    Component for computing various characteristics of the
    structural geometry of the blade.

    parameters
    ----------
    blade_length: float
        physical length of the blade
    blade_surface_st: array
        lofted blade surface with structural discretization normalised to unit
        length
    DP%02d: array
        Arrays of normalized DP curves
    r%02d<materialname>: array
        arrays of material names

    outputs
    -------
    r%02d_thickness: array
        total thickness of each region
    web_angle%02d: array
        angles of webs connecting lower and upper surfaces of OML
    web_offset%02d: array
        offsets in global coordinate system of connections between
        webs and lower and upper surfaces of OML, respectively
    pacc_ss: array
        upper side pitch axis aft cap center in global coordinate system
    pacc_ps: array
        lower side pitch axis aft cap center in global coordinate system
    pacc_ss_curv: array
        curvature of upper side pitch axis aft cap center in
        global coordinate system
    pacc_ps_curv: array
        curvature of lower side pitch axis aft cap center in
        global coordinate system
    thickness_diff: array
        Difference between section thickness
        and material thickness at each DP
    """

    def __init__(self, sdim, st3d, capDPs):
        """
        sdim: tuple
            size of array containing lofted blade surface:
            (chord_ni, span_ni_st, 3).
        st3d: dict
            dictionary containing parametric blade structure.
        capDPs: list
            list of indices of DPs with webs attached to them.
        """
        super(BladeStructureProperties, self).__init__()

        s = st3d['s']
        self.nsec = s.shape[0]
        self.ni_chord = sdim[0]
        self.nDP = st3d['DPs'].shape[1]
        DPs = st3d['DPs']

        # DP indices of webs
        self.web_def = st3d['web_def']
        self.capDPs = capDPs
        self.capDPs.sort()

        self.add_param('blade_length', 1., desc='blade length')
        self.add_param('blade_surface_st', np.zeros(sdim))
        for i in range(self.nDP):
            self.add_param('DP%02d' % i, DPs[:, i])

        self._regions = []
        self._webs = []
        for ireg, reg in enumerate(st3d['regions']):
            layers = []
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                self.add_param(varname + 'T', np.zeros(self.nsec))
                layers.append(varname)
            self._regions.append(layers)
        for ireg, reg in enumerate(st3d['webs']):
            layers = []
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                self.add_param(varname + 'T', np.zeros(self.nsec))
                layers.append(varname)
            self._webs.append(layers)

        for i in range(self.nDP-1):
            self.add_output('r%02d_width' % i, np.zeros(self.nsec),
                            desc='Region%i width' % i)
            self.add_output('r%02d_thickness' % i, np.zeros(self.nsec),
                            desc='Region%i thickness' % i)

        for i, w in enumerate(st3d['web_def']):
            self.add_output('web_angle%02d' % i, np.zeros(self.nsec),
                            desc='Web%02d angle' % i)
            self.add_output('web_offset%02d' % i, np.zeros((self.nsec, 2)),
                            desc='Web%02d offset' % i)

        self.add_output('pacc_ss', np.zeros((self.nsec, 2)),
                        desc='upper side pitch axis aft cap center')
        self.add_output('pacc_ps', np.zeros((self.nsec, 2)),
                        desc='lower side pitch axis aft cap center')
        self.add_output('pacc_ss_curv', np.zeros(self.nsec),
                        desc='upper side pitch axis aft cap center curvature')
        self.add_output('pacc_ps_curv', np.zeros(self.nsec),
                        desc='lower side pitch axis aft cap center curvature')
        self.add_output('thickness_diff', np.zeros((self.nsec, self.nDP/2)),
                        desc='Difference between section thickness'
                             'and material thickness at each DP')

        self.dp_xyz = np.zeros([self.nsec, self.nDP, 3])
        self.dp_s01 = np.zeros([self.nsec, self.nDP])

    def solve_nonlinear(self, params, unknowns, resids):

        smax = np.zeros(self.nsec)
        for i in range(self.nsec):
            x = params['blade_surface_st'][:, i, :]
            af = AirfoilShape(points=x)
            smax[i] = af.smax
            for j in range(self.nDP):
                DP = params['DP%02d' % j][i]
                DPs01 = af.s_to_01(DP)
                self.dp_s01[i, j] = DPs01
                DPxyz = af.interp_s(DPs01)
                self.dp_xyz[i, j, :] = DPxyz

        # upper and lower side pitch axis aft cap center
        unknowns['pacc_ps'][:, :] = (self.dp_xyz[:, self.capDPs[0], [0,1]] + \
                                    self.dp_xyz[:, self.capDPs[1], [0,1]]) / 2.
        unknowns['pacc_ss'][:, :] = (self.dp_xyz[:, self.capDPs[2], [0,1]] + \
                                    self.dp_xyz[:, self.capDPs[3], [0,1]]) / 2.

        # curvatures of region boundary curves
        unknowns['pacc_ps_curv'] = curvature(unknowns['pacc_ps'])
        unknowns['pacc_ss_curv'] = curvature(unknowns['pacc_ss'])

        # web angles and offsets relative to rotor plane
        for i, iw in enumerate(self.web_def):
            offset = self.dp_xyz[:, iw[0], [0,1]] -\
                     self.dp_xyz[:, iw[1], [0,1]]
            angle = -np.array([np.arctan(a) for a in offset[:, 0]/offset[:, 1]]) * 180. / np.pi
            unknowns['web_offset%02d' % i] = offset
            unknowns['web_angle%02d' % i] = angle

        # region widths
        for i in range(self.nDP-1):
            unknowns['r%02d_width' % i] = (self.dp_s01[:, i+1] - self.dp_s01[:, i]) * smax

        # region thicknesses
        for i, reg in enumerate(self._regions):
            t = unknowns['r%02d_thickness' % i]
            t[:] = 0.
            for lname in reg:
                t += np.maximum(0., params[lname + 'T'])

        # compute difference between airfoil thickness and
        # total material thickness at the different DPs
        nr = int(self.nDP/2)
        ri = range(self.nDP-1)
        self.thickness_diff = np.zeros((self.nsec, nr))
        self.shape_thickness = np.zeros((self.nsec, nr))
        # loop DP points from TE to LE
        for i in range(self.nsec):
            for j in range(nr):
                # compute shape thickness assuming symmetric DPs!
                x1, y1 = self.dp_xyz[i, j, :2] * params['blade_length']
                x2, y2 = self.dp_xyz[i, -(j+1), :2] * params['blade_length']
                shape_thickness = ((x1 - x2)**2 + (y1 - y2)**2)**.5
                # material thickness of lower/upper side
                # based on first DP of every region
                t1 = unknowns['r%02d_thickness' % (ri[j])][i]
                t2 = unknowns['r%02d_thickness' % (ri[-(j+1)])][i]
                thick_max = t1 + t2
                self.shape_thickness[i, j] = shape_thickness
                self.thickness_diff[i, j] = shape_thickness - thick_max
        unknowns['thickness_diff'] = self.thickness_diff

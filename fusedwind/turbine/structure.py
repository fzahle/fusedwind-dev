
import time
import re
import numpy as np
from scipy.interpolate import pchip

from openmdao.api import Component, Group, ParallelGroup
from openmdao.core.problem import Problem
from openmdao.api import IndepVarComp

from fusedwind.turbine.geometry import FFDSpline

try:
    from PGL.components.airfoil import AirfoilShape
    from PGL.main.geom_tools import curvature
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

    st3d = {}

    # read mat file
    fid = open(filebase + '.mat', 'r')
    materials = fid.readline().split()[1:]
    st3d['materials'] = {name:i for i, name in enumerate(materials)}
    data = np.loadtxt(fid)
    st3d['matprops'] = data

    # read failmat file
    failcrit = {1:'maximum_strain', 2:'maximum_stress', 3:'tsai_wu'}
    fid = open(filebase + '.failmat', 'r')
    materials = fid.readline().split()[1:]
    data = np.loadtxt(fid)
    st3d['failmat'] = data[:, 1:]
    st3d['failcrit'] = [failcrit[mat] for mat in data[:, 0]]

    # read the dp3d file containing region division points
    dpfile = filebase + '.dp3d'

    dpfid = open(dpfile, 'r')

    # read webs
    wnames = dpfid.readline().split()[1:]
    st3d['web_offsets'] = dpfid.readline().split()[1:]
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

    # read the st3d files containing thicknesses and orientations
    st3d['regions'] = []
    for i, rname in enumerate(regions):
        r = {}
        layup_file = '_'.join([filebase, rname]) + '.st3d'
        fid = open(layup_file, 'r')
        rrname = fid.readline().split()[1]
        lheader = fid.readline().split()[1:]

        cldata = np.loadtxt(fid)
        layers = lheader[1:]
        nl = len(layers)

        # check that layer names are of the type <%s><%02d>
        lnames = []
        basenames = []
        for name in layers:
            try:
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
        rrname = fid.readline().split()[1]
        lheader = fid.readline().split()[1:]

        cldata = np.loadtxt(fid)
        layers = lheader[1:]
        nl = len(layers)

        # check that layer names are of the type <%s><%02d>
        lnames = []
        basenames = []
        for name in layers:
            try:
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
        r['thicknesses'] = cldata[:, 1:nl + 1]
        if cldata.shape[1] == nl*2 + 1:
            r['angles'] = cldata[:, nl + 1:2*nl+1 + 2]
        else:
            r['angles'] = np.zeros((cldata.shape[0], nl))
        st3d['webs'].append(r)

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
    fid.write('# %s\n' % (' '.join(st3d['materials'].keys())))
    fid.write('# E1 E2 E3 nu12 nu13 nu23 G12 G13 G23 rho\n')
    fmt = ' '.join(10*['%.20e'])
    np.savetxt(fid, st3d['matprops'], fmt=fmt)

    failcrit = dict(maximum_strain=1, maximum_stress=2, tsai_wu=3)
    fid = open(filebase + '.failmat', 'w')
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
    webs = ['web%02d' % i for i in range(len(st3d['webs']))]
    fid.write('# %s\n' % ('  '.join(webs)))
    woffsets = st3d['web_offsets']
    fid.write('# %s\n' % ('  '.join(woffsets)))
    for web in st3d['web_def']:
        fid.write('# %i %i\n' % (web[0], web[1]))
    DPs = ['DP%02d' % i for i in range(st3d['DPs'].shape[1])]
    fid.write('# s %s\n' % (' '.join(DPs)))
    data = np.array([st3d['s']]).T
    data = np.append(data, st3d['DPs'], axis=1)
    np.savetxt(fid, data)
    fid.close()

    # write st3d files with material thicknesses and angles
    for i, reg in enumerate(st3d['regions']):
        rname = 'region%02d' % i
        fname = '_'.join([filebase, rname]) + '.st3d'
        fid = open(fname, 'w')
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
    st3dn['s'] = s_new
    st3dn['materials'] = st3d['materials']
    st3dn['matprops'] = st3d['matprops']
    st3dn['failmat'] = st3d['failmat']
    st3dn['failcrit'] = st3d['failcrit']
    st3dn['web_def'] = st3d['web_def']
    st3dn['web_offsets'] = st3d['web_offsets']
    st3dn['regions'] = []
    st3dn['webs'] = []

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

    return st3dn


class SplinedBladeStructure(Group):
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
        super(SplinedBladeStructure, self).__init__()

        self._vars = []
        self._allvars = []
        self.st3dinit = st3d

        # add materials properties array ((10, nmat))
        self.add('matprops_c', IndepVarComp('matprops', st3d['matprops']), promotes=['*'])

        # add materials strength properties array ((18, nmat))
        self.add('failmat_c', IndepVarComp('failmat', st3d['failmat']), promotes=['*'])

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

        st3d = self.st3dinit
        if isinstance(name, str):
            names = [name]
        else:
            names = name
        # decode the name
        if 'DP' in names[0]:
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
                tvars.append(name)
            self._vars.extend(tvars)

            # add the IndepVarComp
            self.add(names[0] + '_c', IndepVarComp(names[0] + '_C', np.zeros(len(Cx))), promotes=['*'])
            for varname in tvars:
                self.connect(names[0] + '_C', varname + '_s.' + varname + '_C')
        else:
            tvars = []

            for name in names:
                if name.startswith('r') or name.startswith('w'):
                    l_index = None
                    # try:
                    ireg = int(name[1:3])
                    try:
                        split = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name[3:], re.I).groups()
                        l_index = split[1]
                    except:
                        split = re.match(r"([a-z]+)([a-z]+)", name[3:], re.I).groups()
                    layername = split[0]+split[1]
                    stype = split[-1]
                    # except:
                    #     raise RuntimeError('Variable name %s not understood' % name)
                else:
                    raise RuntimeError('Variable name %s not understood' % name)

                if name.startswith('r'):
                    r = st3d['regions'][ireg]
                    rname = 'r%02d' % ireg
                elif name.startswith('w'):
                    r = st3d['webs'][ireg]
                    rname = 'w%02d' % ireg

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
                tvars.append(name)

            self._vars.extend(tvars)
            # finally add the IndepVarComp and make the connections
            self.add(names[0] + '_c', IndepVarComp(names[0] + '_C', np.zeros(len(Cx))), promotes=['*'])
            for varname in tvars:
                self.connect(names[0] + '_C', varname + '_s.' + varname + '_C')

    def configure(self):
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
    pacc_u: array
        upper side pitch axis aft cap center in global coordinate system
    pacc_l: array
        lower side pitch axis aft cap center in global coordinate system
    pacc_u_curv: array
        curvature of upper side pitch axis aft cap center in
        global coordinate system
    pacc_l_curv: array
        curvature of lower side pitch axis aft cap center in
        global coordinate system
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

        self.add_param('blade_length', 1., units='m', desc='blade length')
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
            self.add_output('r%02d_width' % i, np.zeros(self.nsec), desc='Region%i width' % i)
            self.add_output('r%02d_thickness' % i, np.zeros(self.nsec), desc='Region%i thickness' % i)

        for i, w in enumerate(st3d['web_def']):
            self.add_output('web_angle%02d' % i, np.zeros(self.nsec), desc='Web%02d angle' % i)
            self.add_output('web_offset%02d' % i, np.zeros((self.nsec, 2)), desc='Web%02d offset' % i)

        self.add_output('pacc_u', np.zeros((self.nsec, 2)), desc='upper side pitch axis aft cap center')
        self.add_output('pacc_l', np.zeros((self.nsec, 2)), desc='lower side pitch axis aft cap center')
        self.add_output('pacc_u_curv', np.zeros(self.nsec), desc='upper side pitch axis aft cap center curvature')
        self.add_output('pacc_l_curv', np.zeros(self.nsec), desc='lower side pitch axis aft cap center curvature')

        self.dp_xyz = np.zeros([self.nsec, self.nDP, 3])
        self.dp_s01 = np.zeros([self.nsec, self.nDP])

    def solve_nonlinear(self, params, unknowns, resids):

        smax = np.zeros(self.nsec)
        for i in range(self.nsec):
            x = params['blade_surface_st'][:, i, :]
            x[:, 2] *= params['blade_length']
            af = AirfoilShape(points=x)
            smax[i] = af.smax
            for j in range(self.nDP):
                DP = params['DP%02d' % j][i]
                DPs01 = af.s_to_01(DP)
                self.dp_s01[i, j] = DPs01
                DPxyz = af.interp_s(DPs01)
                self.dp_xyz[i, j, :] = DPxyz

        # upper and lower side pitch axis aft cap center
        unknowns['pacc_l'][:, :] = (self.dp_xyz[:, self.capDPs[0], [0,1]] + \
                                    self.dp_xyz[:, self.capDPs[1], [0,1]]) / 2.
        unknowns['pacc_u'][:, :] = (self.dp_xyz[:, self.capDPs[2], [0,1]] + \
                                    self.dp_xyz[:, self.capDPs[3], [0,1]]) / 2.

        # curvatures of region boundary curves
        unknowns['pacc_l_curv'] = curvature(unknowns['pacc_l'])
        unknowns['pacc_u_curv'] = curvature(unknowns['pacc_u'])

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
                t += params[lname + 'T']

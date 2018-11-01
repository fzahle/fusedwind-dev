
import numpy as np
from scipy.interpolate import pchip, Akima1DInterpolator
from scipy.linalg import norm

from openmdao.api import Component, Group, IndepVarComp
from openmdao.util.options import OptionsDictionary

from fusedwind.lib.geom_tools import calculate_length, curvature
from fusedwind.lib.naturalcubicspline import NaturalCubicSpline

try:
    from PGL.main.planform import redistribute_planform
    from PGL.components.loftedblade import LoftedBladeSurface
    from PGL.main.bezier import BezierCurve
    _PGL_installed = True
except:
    _PGL_installed = False
    print 'warning: PGL not installed'


class SplineBase(object):
    """
    base for 1-D splines

    if the spline requires it, implement a fitting procedure in __init__

    place the main call to the spline in __call__
    """

    def initialize(self, Cx, xp, yp):

        pass

    def normdist(self, xp):
        """normalize x distribution"""

        return (xp - xp[0]) / (xp[-1] - xp[0])

    def __call__(self, x, Cx, C):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with x-coordinates of spline control points
        yp: array
            array with y-coordinates of spline control points

        returns
        ---------
        ynew: array
            resampled points
        """

        raise NotImplementedError('A derived class of SplineBase needs to implement a __call__ method')


class LinearSpline(SplineBase):

    def initialize(self, x, xp, yp):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with original x-distribution
        yp: array
            array with original y-distribution

        returns
        ---------
        ynew: array
            resampled points
        """

        return self.__call__(x, xp, yp)

    def __call__(self, x, Cx, C):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with x-coordinates of spline control points
        yp: array
            array with y-coordinates of spline control points

        returns
        ---------
        ynew: array
            resampled points
        """

        return np.interp(x, Cx, C)


class pchipSpline(SplineBase):

    def initialize(self, x, xp, yp):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with original x-distribution
        yp: array
            array with original y-distribution

        returns
        ---------
        ynew: array
            resampled points
        """

        return self.__call__(x, xp, yp)

    def __call__(self, x, Cx, C):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with x-coordinates of spline control points
        yp: array
            array with y-coordinates of spline control points

        returns
        ---------
        ynew: array
            resampled points
        """

        spl = pchip(Cx, C)
        return spl(x)


class BezierSpline(SplineBase):

    def initialize(self, x, xp, yp):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with original x-distribution
        yp: array
            array with original y-distribution

        returns
        ---------
        ynew: array
            resampled points
        """
        self.B = BezierCurve()
        self.B.CPs = np.array([xp, yp]).T
        return self.__call__(x, xp, yp)

    def __call__(self, x, Cx, C):
        """
        params:
        ----------
        x: array
            array with new x-distribution
        xp: array
            array with x-coordinates of spline control points
        yp: array
            array with y-coordinates of spline control points

        returns
        ---------
        ynew: array
            resampled points
        """
        self.B.CPs = np.array([Cx, C]).T
        self.B.update()
        spl = NaturalCubicSpline(self.B.points[:, 0], self.B.points[:, 1])
        return spl(x)


spline_dict = {'linear': LinearSpline,
               'pchip': pchipSpline,
               'bezier': BezierSpline}


def read_blade_planform(filename):
    """
    read a planform file with columns:

    |  s: normalized running length of blade
    |  x: x-coordinates of blade axis
    |  y: y-coordinates of blade axis
    |  z: z-coordinates of blade axis
    |  rot_x: x-rotation of blade axis
    |  rot_y: y-rotation of blade axis
    |  rot_z: z-rotation of blade axis
    |  chord: chord distribution
    |  rthick: relative thickness distribution
    |  p_le: pitch axis aft leading edge distribution
    |  dy: vertical offset of cross-section

    parameters
    ----------
    filename: str
        path to file containing planform data

    returns
    -------
    pf: dict
        dictionary containing planform data normalized
        to a span of 1.
    """

    data = np.loadtxt(filename)
    s = calculate_length(data[:, [0, 1, 2]])

    pf = {}
    pf['blade_length'] = data[-1, 2]
    pf['s'] = s / s[-1]
    pf['smax'] = s[-1]
    pf['x'] = data[:, 0] / data[-1, 2]
    pf['y'] = data[:, 1] / data[-1, 2]
    pf['z'] = data[:, 2] / data[-1, 2]
    pf['rot_x'] = data[:, 3]
    pf['rot_y'] = data[:, 4]
    pf['rot_z'] = data[:, 5]
    pf['chord'] = data[:, 6] / data[-1, 2]
    pf['rthick'] = data[:, 7]
    pf['rthick'] /= pf['rthick'].max()
    pf['athick'] = pf['rthick'] * pf['chord']
    pf['p_le'] = data[:, 8]
    try:
        pf['dy'] = data[:, 9]
    except:
        pf['dy'] = np.zeros(data.shape[0])

    return pf

def write_blade_planform(pf, filename):
    """
    write a planform file with columns:

    |  s: normalized running length of blade
    |  x: x-coordinates of blade axis
    |  y: y-coordinates of blade axis
    |  z: z-coordinates of blade axis
    |  rot_x: x-rotation of blade axis
    |  rot_y: y-rotation of blade axis
    |  rot_z: z-rotation of blade axis
    |  chord: chord distribution
    |  rthick: relative thickness distribution
    |  p_le: pitch axis aft leading edge distribution
    |  dy: vertical offset of cross-section

    parameters
    ----------
    pf: dict
        planform dictionary
    filename: str
        path to file containing planform data
    """

    data = np.zeros((pf['x'].shape[0], 10))
    s = calculate_length(data[:, [0, 1, 2]])

    names = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z',
             'chord', 'rthick', 'p_le', 'dy']
    for i, name in enumerate(names):
        try:
            data[:, i] = pf[name]
        except:
            print 'failed writing %s - assuming zeros' % name
            data[:, i] = np.zeros(s.shape[0])
    fid = open(filename, 'w')
    exp_prec = 15             # exponential precesion
    col_width = exp_prec + 10  # column width required for exp precision
    header_full = '#'
    header_full += ''.join([(hh + ' [%i]').center(col_width + 1) % i
                           for i, hh in enumerate(names)]) + '\n'
    fid.write(header_full)
    np.savetxt(fid, data)


class BladePlanformWriter(Component):

    def __init__(self, size_in, filebase='blade'):
        super(BladePlanformWriter, self).__init__()

        self.filebase = filebase + '%i' % self.__hash__()

        self.add_param('x', np.zeros(size_in))
        self.add_param('y', np.zeros(size_in))
        self.add_param('z', np.zeros(size_in))
        self.add_param('chord', np.zeros(size_in))
        self.add_param('rthick', np.zeros(size_in))
        self.add_param('rot_x', np.zeros(size_in))
        self.add_param('rot_y', np.zeros(size_in))
        self.add_param('rot_z', np.zeros(size_in))
        self.add_param('p_le', np.zeros(size_in))
        self.add_param('dy', np.zeros(size_in))

        self._exec_count = 0

    def solve_nonlinear(self, params, unknowns, resids):

        self._exec_count += 1

        pf = {}
        pf['x'] = params['x']
        pf['y'] = params['y']
        pf['z'] = params['z']
        pf['rot_x'] = params['rot_x']
        pf['rot_y'] = params['rot_y']
        pf['rot_z'] = params['rot_z']
        pf['chord'] = params['chord']
        pf['rthick'] = params['rthick']
        pf['p_le'] = params['p_le']
        pf['dy'] = params['dy']

        write_blade_planform(pf, self.filebase + '_it%i.pfd'%self._exec_count)


def redistribute_planform(pf, dist=[], s=None, spline_type='akima'):
    """
    redistribute a blade planform

    calls PGL.main.planform.read_blade_planform

    parameters
    ----------
    pf: dict
        optional dictionary containing planform. If not supplied, planform_filename is required. Keys:

        |  s: normalized running length of blade
        |  x: x-coordinates of blade axis
        |  y: y-coordinates of blade axis
        |  z: z-coordinates of blade axis
        |  rot_x: x-rotation of blade axis
        |  rot_y: y-rotation of blade axis
        |  rot_z: z-rotation of blade axis
        |  chord: chord distribution
        |  rthick: relative thickness distribution
        |  p_le: pitch axis aft leading edge distribution
        |  dy: vertical offset of cross-section
    dist: list
        list of control points with the form

        | [[s0, ds0, n0], [s1, ds1, n1], ... [s<n>, ds<n>, n<n>]]

        | where

            | s<n> is the curve fraction at each control point,
            | ds<n> is the cell size at each control point,
            | n<n> is the cell count at each control point.
    s: array
        optional normalized distribution of cells.
    """

    from PGL.main.planform import redistribute_planform
    return redistribute_planform(pf, dist, s, spline_type)


class PGLRedistributedPlanform(Component):
    """
    simple component for redistributing a planform
    using PGL.main.planform.redistribute_planform

    parameters
    ----------
    s: array
        normalized running length of blade
    x: array
        x-coordinates of blade axis
    y: array
        y-coordinates of blade axis
    z: array
        z-coordinates of blade axis
    rot_x: array
        x-rotation of blade axis
    rot_y: array
        y-rotation of blade axis
    rot_z: array
        z-rotation of blade axis
    chord: array
        chord distribution
    rthick: array
        relative thickness distribution
    p_le: array
        pitch axis aft leading edge distribution
    dy: array
        vertical offset of cross-section
    """

    def __init__(self, name, size_in, s_new):
        """
        parameters
        ----------
        name: str
            name appended to output planform
        size_in: int
            size of input planform
        s_new: array
            distribution of output planform
        """
        super(PGLRedistributedPlanform, self).__init__()

        # options are 'linear', 'ncubic', 'pchip', 'akima'
        self.spline_type = 'akima'

        self.add_param('s', np.zeros(size_in))
        self.add_param('x', np.zeros(size_in))
        self.add_param('y', np.zeros(size_in))
        self.add_param('z', np.zeros(size_in))
        self.add_param('chord', np.zeros(size_in))
        self.add_param('rthick', np.zeros(size_in))
        self.add_param('rot_x', np.zeros(size_in))
        self.add_param('rot_y', np.zeros(size_in))
        self.add_param('rot_z', np.zeros(size_in))
        self.add_param('p_le', np.zeros(size_in))
        self.add_param('dy', np.zeros(size_in))

        self.s_new = s_new
        size_out = s_new.shape[0]
        self._suffix = name
        self.add_output('s'+name, np.zeros(size_out))
        self.add_output('x'+name, np.zeros(size_out))
        self.add_output('y'+name, np.zeros(size_out))
        self.add_output('z'+name, np.zeros(size_out))
        self.add_output('chord'+name, np.zeros(size_out))
        self.add_output('rthick'+name, np.zeros(size_out))
        self.add_output('rot_x'+name, np.zeros(size_out))
        self.add_output('rot_y'+name, np.zeros(size_out))
        self.add_output('rot_z'+name, np.zeros(size_out))
        self.add_output('p_le'+name, np.zeros(size_out))
        self.add_output('dy'+name, np.zeros(size_out))
        self.add_output('athick'+name, np.zeros(size_out))


    def solve_nonlinear(self, params, unknowns, resids):

        # we need to dig into the _ByObjWrapper val to get the array
        # values out
        # pf_in = {name: val['val'].val for name, val in params.iteritems()}
        pf_in = {}
        pf_in['s'] = params['s']
        pf_in['x'] = params['x']
        pf_in['y'] = params['y']
        pf_in['z'] = params['z']
        pf_in['rot_x'] = params['rot_x']
        pf_in['rot_y'] = params['rot_y']
        pf_in['rot_z'] = params['rot_z']
        pf_in['chord'] = params['chord']
        pf_in['rthick'] = params['rthick']
        pf_in['p_le'] = params['p_le']
        pf_in['dy'] = params['dy']

        if _PGL_installed:
            pf = redistribute_planform(pf_in, s=self.s_new, spline_type=self.spline_type)
        else:
            pf = {}
            for k, v in pf_in.iteritems():
                spl = pchip(pf_in['s'], v)
                pf[k] = spl(self.s_new)

        for k, v in pf.iteritems():
            unknowns[k+self._suffix] = v
        unknowns['athick'+self._suffix] = pf['chord'] * pf['rthick']


class FFDSpline(Component):
    """
    Spline that deforms a base shape using a choice of spline function
    """


    def __init__(self, name, s, P, Cx, scaler=1.):
        super(FFDSpline, self).__init__()

        self._name = name

        opt = self.spline_options = OptionsDictionary()
        opt.add_option('spline_type', 'bezier', values=['linear',
                                                        'pchip',
                                                        'bezier'],\
                       desc='spline type used in FFD')
        self.nC = Cx.shape[0]
        self.Cx = Cx
        self.s = s
        self._iCx0 = 0
        self._iCx1 = None
        if self.Cx[0] > 0.:
            self._iCx0 = np.where(np.abs(self.s-self.Cx[0])==np.abs(self.s-self.Cx[0]).min())[0]
            if isinstance(self._iCx0, np.ndarray):
                self._iCx0 = self._iCx0[0]
        if self.Cx[-1] < 1.:
            self._iCx1 = np.where(np.abs(self.s-self.Cx[-1])==np.abs(self.s-self.Cx[-1]).min())[0]+1
            if isinstance(self._iCx1, np.ndarray):
                self._iCx1 = self._iCx1[0]
        self.Pinit = P
        self._size = P.shape[0]
        self.scaler = scaler

        self.add_param(name + '_C', np.zeros(self.nC), desc='spline control points')
        self.add_output(name, np.zeros(self._size))
        self.add_output(name + '_curv', np.zeros(self._size))

        self._init_called = False
        self.spline = None

        self.set_spline(self.spline_options['spline_type'])

    def set_spline(self, spline_type):

        self.spline = spline_dict[spline_type]()
        self.spline_options['spline_type'] = spline_type

    def solve_nonlinear(self, params, unknowns, resids):
        """
        update the spline
        """

        C = params[self._name + '_C']

        if not self._init_called:
            self.set_spline(self.spline_options['spline_type'])
            self.spline.initialize(self.s[self._iCx0:self._iCx1], self.Cx, C)
        self._P = self.spline(self.s[self._iCx0:self._iCx1], self.Cx, C)
        _P = np.zeros(self.Pinit.shape[0])
        _P[self._iCx0:self._iCx1] = self._P

        P = self.Pinit + _P * self.scaler
        curv = curvature(np.array([self.s, P]).T)
        unknowns[self._name] = P
        unknowns[self._name + '_curv'] = curv


class ScaleChord(Component):
    """
    component for scaling chord with 1./blade_length
    """

    def __init__(self, size, suffix=''):
        super(ScaleChord, self).__init__()

        self.add_param('blade_scale', 1.)
        self.add_param('chord_in', np.zeros(size))
        self.add_output('chord' + suffix, np.zeros(size))
        self._suffix = suffix

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['chord' + self._suffix] = params['chord_in'] / params['blade_scale']


class ComputeAthick(Component):
    """
    component to replace connection:
    connect('chord.P*rthick.P', 'pfOut.athick')
    """

    def __init__(self, size):
        super(ComputeAthick, self).__init__()

        self.add_param('chord', np.zeros(size))
        self.add_param('rthick', np.zeros(size))
        self.add_output('athick', np.zeros(size))

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['athick'] = params['chord'] * params['rthick']


class ComputeSmax(Component):

    def __init__(self, pf):
        super(ComputeSmax, self).__init__()

        self.add_param('x', pf['x'])
        self.add_param('y', pf['y'])
        self.add_param('z', pf['z'])
        self.add_output('blade_curve_length', 0.)
        self.add_output('s', np.zeros(pf['x'].shape[0]))

    def solve_nonlinear(self, params, unknowns, resids):

        s = calculate_length(np.array([params['x'],
                                       params['y'],
                                       params['z']]).T)
        unknowns['blade_curve_length'] = s[-1]
        unknowns['s'] = s


class SplinedBladePlanform(Group):
    """
    Class that adds planform variables to the analysis
    either as splines with user defined control points
    or according to the initial planform data
    """

    def __init__(self, pf):
        """
        parameters
        ----------
        pf: dict
            dictionary containing planform with required spanwise resolution.
            Keys:
            |  s: normalized running length of blade
            |  x: x-coordinates of blade axis
            |  y: y-coordinates of blade axis
            |  z: z-coordinates of blade axis
            |  rot_x: x-rotation of blade axis
            |  rot_y: y-rotation of blade axis
            |  rot_z: z-rotation of blade axis
            |  chord: chord distribution
            |  rthick: relative thickness distribution
            |  p_le: pitch axis aft leading edge distribution
            |  dy: vertical offset of cross-section

        """
        super(SplinedBladePlanform, self).__init__()

        self._size = pf['s'].shape[0]
        self.pfinit = pf
        self._vars = []

    def add_spline(self, name, Cx, spline_type='bezier', scaler=1.):
        """
        adds an FFDSpline for the given planform variable
        with user defined spline type and control point locations

        parameters
        ----------
        name: str
            name of planform variable. Options:
            |  x: x-coordinates of blade axis
            |  y: y-coordinates of blade axis
            |  rot_x: x-rotation of blade axis
            |  rot_y: y-rotation of blade axis
            |  rot_z: z-rotation of blade axis
            |  chord: chord distribution
            |  rthick: relative thickness distribution
            |  p_le: pitch axis aft leading edge distribution
            |  dy: vertical offset of cross-section
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip
        """
        if name not in ['x', 'y', 'rot_x', 'rot_y', 'rot_z', 'chord', 'rthick', 'p_le', 'dy']:
            raise RuntimeError('%s not in planform dictionary' % name)


        self._vars.append(name)
        # chord needs to be scaled according to blade scale parameter
        spl_indeps = []
        if name == 'chord':
            cname = name + '_c'
            self.add(cname, IndepVarComp(name + '_C', np.zeros(len(Cx))), promotes=['*'])
            c = self.add(name + '_s', FFDSpline('chord',
                                                s=self.pfinit['s'],
                                                P=self.pfinit['chord'],
                                                Cx=Cx, scaler=scaler),
                                                promotes=['chord_C', 'chord_curv'])
            c.spline_options['spline_type'] = spline_type
            c.set_spline(spline_type)
            self.add('chord_scaler', ScaleChord(self._size), promotes=['blade_scale', 'chord'])
            self.connect('chord_s.chord', 'chord_scaler.chord_in')
        else:
            cname = name + '_c'
            self.add(cname, IndepVarComp(name + '_C', np.zeros(len(Cx))), promotes=['*'])
            c = self.add(name + '_s', FFDSpline(name,
                                                s=self.pfinit['s'],
                                                P=self.pfinit[name],
                                                Cx=Cx, scaler=scaler),
                                                promotes=[name, name + '_C', name + '_curv'])
            c.spline_options['spline_type'] = spline_type
            c.set_spline(spline_type)

    def configure(self):

        print 'SplinedBladePlanform: No harm done, but configure is depreciated\n' + \
              'and replaced by pre_setup called automatically by OpenMDAO.\n'+\
              'Ensure that you have OpenMDAO > v1.7.1 installed'

    def pre_setup(self, problem):
        """
        add IndepVarComp's for all remaining planform variables
        """
        indeps = list(set(['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z', 'chord', 'rthick', 'p_le', 'dy'])-set(self._vars))

        indep_tups = []
        for name in indeps:
            indep_tups.append((name, self.pfinit[name]))
        self.add('pf_c', IndepVarComp(indep_tups), promotes=indeps)


        c = self.add('smax_c', ComputeSmax(self.pfinit), promotes=['blade_curve_length', 's'])
        self.connect('x', 'smax_c.x')
        self.connect('y', 'smax_c.y')
        self.connect('z', 'smax_c.z')
        self.add('athick_c', ComputeAthick(self._size), promotes=['athick'])
        self.connect('rthick', 'athick_c.rthick')
        self.connect('chord', 'athick_c.chord')


class PGLLoftedBladeSurface(Component):
    """
    class for generating a simple lofted blade surface
    based on a series of base airfoils
    and a planform definition using
    PGL.components.loftedblade.LoftedBladeSurface
    """


    def __init__(self, config, size_in=200, size_out=(200, 20, 3), suffix=''):
        super(PGLLoftedBladeSurface, self).__init__()

        self._dry_run = False
        if not _PGL_installed:
            self._dry_run = True

        self.add_param('blade_length', 1.)

        names = ['s', 'x', 'y', 'z',
                 'rot_x', 'rot_y', 'rot_z',
                 'chord', 'rthick','p_le', 'dy']
        for name in names:
            self.add_param(name+suffix, np.zeros(size_in))

        self._suffix = suffix
        self.add_output('blade_surface' + suffix, np.zeros(size_out))
        self.add_output('blade_surface_norm' + suffix, np.zeros(size_out))

        # for i in range(size_in[1]):
        #     self.add_param('base_af%02d' % i, np.zeros(size_in[0], 2))

        # configuration variables for PGL's LoftedBladeSurface class
        self.config = {}
        self.config['base_airfoils'] = []
        self.config['blend_var'] = np.array([])
        self.config['user_surface'] = np.array([])
        self.config['user_surface_file'] = ''
        self.config['user_surface_shape'] = ()
        self.config['ni_chord'] = size_out[0]
        self.config['chord_nte'] = 0
        self.config['redistribute_flag'] = False
        self.config['x_chordwise'] = np.array([])
        self.config['minTE'] = 0.
        self.config['interp_type'] = 'rthick'
        self.config['surface_spline'] = 'pchip'
        self.config['dist_LE'] = np.array([])
        self.config['gf_heights'] = np.array([])

        for k, v in config.iteritems():
            if k in self.config.keys():
                self.config[k] = v
            else:
                print 'unknown config key %s' % k

        self.rot_order = np.array([2,1,0])

        if not self._dry_run:
            self.pgl_surf = LoftedBladeSurface(**self.config)
        self._pgl_config_called = False

    def _configure_interpolator(self):

        if self.config['base_airfoils'] == 0:
            raise RuntimeError('base_airfoils list is empty')
        if self.config['blend_var'].shape[0] == 0:
            raise RuntimeError('blend_var array is empty')
        self.pgl_surf.ni_chord = self.config['ni_chord']
        self.pgl_surf.surface_spline = self.config['surface_spline']
        self.pgl_surf.blend_var = self.config['blend_var']
        self.pgl_surf.base_airfoils = self.config['base_airfoils']
        self.pgl_surf.initialize_interpolator()
        self._pgl_config_called = True

    def solve_nonlinear(self, params, unknowns, resids):

        if self._dry_run:
            ni = params['s'  + self._suffix].shape[0]
            surf = np.zeros([self.config['ni_chord'], ni, 3])
            surf[:, :, 2] = params['s'  + self._suffix]
            self.unknowns['blade_surface' + self._suffix] = surf
            self.unknowns['blade_surface_norm' + self._suffix] = surf
            return

        if not self._pgl_config_called:
            self._configure_interpolator()
        # we need to dig into the _ByObjWrapper val to get the array
        # values out
        pf = {}
        pf['s'] = params['s' + self._suffix]
        pf['x'] = params['x' + self._suffix]
        pf['y'] = params['y' + self._suffix]
        pf['z'] = params['z' + self._suffix]
        pf['rot_x'] = params['rot_x' + self._suffix]
        pf['rot_y'] = params['rot_y' + self._suffix]
        pf['rot_z'] = params['rot_z' + self._suffix]
        pf['chord'] = params['chord' + self._suffix]
        pf['rthick'] = params['rthick' + self._suffix]
        pf['p_le'] = params['p_le' + self._suffix]
        pf['dy'] = params['dy' + self._suffix]
        self.pgl_surf.pf = pf
        self.pgl_surf.build_blade()

        surf = self.pgl_surf.surface
        surfnorot = self.pgl_surf.surfnorot

        self.unknowns['blade_surface' + self._suffix] = surf
        self.unknowns['blade_surface_norm' + self._suffix] = surfnorot

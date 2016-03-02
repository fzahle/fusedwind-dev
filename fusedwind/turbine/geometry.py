
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


spline_dict = {'pchip': pchipSpline,
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

    parameters
    ----------
    pf: dict
        planform dictionary
    filename: str
        path to file containing planform data
    """

    data = np.zeros((pf['x'].shape[0], 9))
    s = calculate_length(data[:, [0, 1, 2]])

    names = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z',
             'chord', 'rthick', 'p_le']
    for i, name in enumerate(names):
        data[:, i] = pf[name]
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

        self.add_param('blade_ae:x', np.zeros(size_in))
        self.add_param('blade_ae:y', np.zeros(size_in))
        self.add_param('blade_ae:z', np.zeros(size_in))
        self.add_param('blade_ae:chord', np.zeros(size_in))
        self.add_param('blade_ae:rthick', np.zeros(size_in))
        self.add_param('blade_ae:rot_x', np.zeros(size_in))
        self.add_param('blade_ae:rot_y', np.zeros(size_in))
        self.add_param('blade_ae:rot_z', np.zeros(size_in))
        self.add_param('blade_ae:p_le', np.zeros(size_in))

        self._exec_count = 0

    def solve_nonlinear(self, params, unknowns, resids):

        self._exec_count += 1

        pf = {}
        pf['x'] = params['blade_ae:x']
        pf['y'] = params['blade_ae:y']
        pf['z'] = params['blade_ae:z']
        pf['rot_x'] = params['blade_ae:rot_x']
        pf['rot_y'] = params['blade_ae:rot_y']
        pf['rot_z'] = params['blade_ae:rot_z']
        pf['chord'] = params['blade_ae:chord']
        pf['rthick'] = params['blade_ae:rthick']
        pf['p_le'] = params['blade_ae:p_le']

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

        self.add_param('blade_ae:s', np.zeros(size_in))
        self.add_param('blade_ae:x', np.zeros(size_in))
        self.add_param('blade_ae:y', np.zeros(size_in))
        self.add_param('blade_ae:z', np.zeros(size_in))
        self.add_param('blade_ae:chord', np.zeros(size_in))
        self.add_param('blade_ae:rthick', np.zeros(size_in))
        self.add_param('blade_ae:rot_x', np.zeros(size_in))
        self.add_param('blade_ae:rot_y', np.zeros(size_in))
        self.add_param('blade_ae:rot_z', np.zeros(size_in))
        self.add_param('blade_ae:p_le', np.zeros(size_in))

        self.s_new = s_new
        size_out = s_new.shape[0]
        self._suffix = name
        self.add_output('blade%s:s' % name, np.zeros(size_out))
        self.add_output('blade%s:x' % name, np.zeros(size_out))
        self.add_output('blade%s:y' % name, np.zeros(size_out))
        self.add_output('blade%s:z' % name, np.zeros(size_out))
        self.add_output('blade%s:chord' % name, np.zeros(size_out))
        self.add_output('blade%s:rthick' % name, np.zeros(size_out))
        self.add_output('blade%s:rot_x' % name, np.zeros(size_out))
        self.add_output('blade%s:rot_y' % name, np.zeros(size_out))
        self.add_output('blade%s:rot_z' % name, np.zeros(size_out))
        self.add_output('blade%s:p_le' % name, np.zeros(size_out))
        self.add_output('blade%s:athick' % name, np.zeros(size_out))


    def solve_nonlinear(self, params, unknowns, resids):

        pf_in = {}
        pf_in['s'] = params['blade_ae:s']
        pf_in['x'] = params['blade_ae:x']
        pf_in['y'] = params['blade_ae:y']
        pf_in['z'] = params['blade_ae:z']
        pf_in['rot_x'] = params['blade_ae:rot_x']
        pf_in['rot_y'] = params['blade_ae:rot_y']
        pf_in['rot_z'] = params['blade_ae:rot_z']
        pf_in['chord'] = params['blade_ae:chord']
        pf_in['rthick'] = params['blade_ae:rthick']
        pf_in['p_le'] = params['blade_ae:p_le']

        if _PGL_installed:
            pf = redistribute_planform(pf_in, s=self.s_new, spline_type=self.spline_type)
        else:
            pf = {}
            for k, v in pf_in.iteritems():
                spl = pchip(pf_in['s'], v)
                pf[k] = spl(self.s_new)

        for k, v in pf.iteritems():
            unknowns['blade%s:%s' % (self._suffix, k)] = v
        unknowns['blade%s:athick' % self._suffix] = pf['chord'] * pf['rthick']


class FFDSpline(Component):
    """
    Spline that deforms a base shape using a choice of spline function
    """


    def __init__(self, name, s, P, Cx, scaler=1.):
        super(FFDSpline, self).__init__()

        self._name = name

        opt = self.spline_options = OptionsDictionary()
        opt.add_option('spline_type', 'bezier', values=['pchip', 'bezier'],\
                       desc='spline type used in FFD')
        self.nC = Cx.shape[0]
        self.Cx = Cx
        self.s = s
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
            # self.Pbase = self.base_spline(self.s, self.xinit, self.Pinit)
            self.spline.initialize(self.s, self.Cx, C)
        self._P = self.spline(self.s, self.Cx, C)
        P = self.Pinit + self._P * self.scaler
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
        self.add_param('blade_ae:chord_in', np.zeros(size))
        self.add_output('blade_ae:chord' + suffix, np.zeros(size))
        self._suffix = suffix

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['blade_ae:chord' + self._suffix] = params['blade_ae:chord_in'] / params['blade_scale']


class ComputeAthick(Component):
    """
    component to replace connection:
    connect('chord.P*rthick.P', 'pfOut.athick')
    """

    def __init__(self, size):
        super(ComputeAthick, self).__init__()

        self.add_param('blade_ae:chord', np.zeros(size))
        self.add_param('blade_ae:rthick', np.zeros(size))
        self.add_output('blade_ae:athick', np.zeros(size))

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['blade_ae:athick'] = params['blade_ae:chord'] * params['blade_ae:rthick']


class ComputeSmax(Component):

    def __init__(self, pf):
        super(ComputeSmax, self).__init__()

        self.add_param('blade_ae:x', pf['x'])
        self.add_param('blade_ae:y', pf['y'])
        self.add_param('blade_ae:z', pf['z'])
        self.add_output('blade_curve_length', 0.)

    def solve_nonlinear(self, params, unknowns, resids):

        s = calculate_length(np.array([params['blade_ae:x'],
                                       params['blade_ae:y'],
                                       params['blade_ae:z']]).T)
        unknowns['blade_curve_length'] = s[-1]


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
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip
        """
        if name not in ['x', 'y', 'rot_x', 'rot_y', 'rot_z', 'chord', 'rthick', 'p_le']:
            raise RuntimeError('%s not in planform dictionary' % name)


        self._vars.append(name)
        # chord needs to be scaled according to blade scale parameter
        if name == 'chord':
            cname = name + '_c'
            self.add(cname, IndepVarComp('blade_ae:'+name + '_C', np.zeros(len(Cx))), promotes=['*'])
            c = self.add(name + '_s', FFDSpline('blade_ae:chord',
                                                s=self.pfinit['s'],
                                                P=self.pfinit['chord'],
                                                Cx=Cx, scaler=scaler),
                                                promotes=['blade_ae:chord_C'])
            c.spline_options['spline_type'] = spline_type
            self.add('chord_scaler', ScaleChord(self._size), promotes=['blade_scale', 'blade_ae:chord'])
            self.connect('chord_s.blade_ae:chord', 'chord_scaler.blade_ae:chord_in')
        else:
            cname = name + '_c'
            self.add(cname, IndepVarComp('blade_ae:'+name + '_C', np.zeros(len(Cx))), promotes=['*'])
            c = self.add(name + '_s', FFDSpline('blade_ae:' + name,
                                                s=self.pfinit['s'],
                                                P=self.pfinit[name],
                                                Cx=Cx, scaler=scaler),
                                                promotes=['blade_ae:' + name,
                                                          'blade_ae:' + name + '_C'])
            c.spline_options['spline_type'] = spline_type

    def configure(self):
        """
        add IndepVarComp's for all remaining planform variables
        """
        indeps = list(set(['s', 'x', 'y', 'z',
                           'rot_x', 'rot_y', 'rot_z',
                           'chord', 'rthick', 'p_le']) -
                            set(self._vars))

        for name in indeps:
            self.add(name+'_c', IndepVarComp('blade_ae:'+name, self.pfinit[name]), promotes=['blade_ae:'+name])


        c = self.add('smax_c', ComputeSmax(self.pfinit), promotes=['blade_curve_length'])
        self.connect('blade_ae:x', 'smax_c.blade_ae:x')
        self.connect('blade_ae:y', 'smax_c.blade_ae:y')
        self.connect('blade_ae:z', 'smax_c.blade_ae:z')
        self.add('athick_c', ComputeAthick(self._size), promotes=['blade_ae:athick'])
        self.connect('blade_ae:rthick', 'athick_c.blade_ae:rthick')
        self.connect('blade_ae:chord', 'athick_c.blade_ae:chord')


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
                 'chord', 'rthick','p_le']
        for name in names:
            self.add_param('blade%s:%s' % (suffix, name), np.zeros(size_in))

        self._suffix = suffix
        self.add_output('blade%s:surface' % suffix, np.zeros(size_out))
        self.add_output('blade%s:surface_norm' % suffix, np.zeros(size_out))

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
            ni = params['blade%s:s' % self._suffix].shape[0]
            surf = np.zeros([self.config['ni_chord'], ni, 3])
            surf[:, :, 2] = params['blade%s:s' % self._suffix]
            self.unknowns['blade%s:surface' % self._suffix] = surf
            self.unknowns['blade%s:surface_norm' % self._suffix] = surf
            return

        if not self._pgl_config_called:
            self._configure_interpolator()
        # we need to dig into the _ByObjWrapper val to get the array
        # values out
        pf = {}
        pf['s'] = params['blade%s:s' % self._suffix]
        pf['x'] = params['blade%s:x' % self._suffix]
        pf['y'] = params['blade%s:y' % self._suffix]
        pf['z'] = params['blade%s:z' % self._suffix]
        pf['rot_x'] = params['blade%s:rot_x' % self._suffix]
        pf['rot_y'] = params['blade%s:rot_y' % self._suffix]
        pf['rot_z'] = params['blade%s:rot_z' % self._suffix]
        pf['chord'] = params['blade%s:chord' % self._suffix]
        pf['rthick'] = params['blade%s:rthick' % self._suffix]
        pf['p_le'] = params['blade%s:p_le' % self._suffix]

        # call PGL to build lofted surface
        self.pgl_surf.pf = pf
        self.pgl_surf.build_blade()

        surf = self.pgl_surf.surface
        surfnorot = self.pgl_surf.surfnorot

        self.unknowns['blade%s:surface' % self._suffix] = surf
        self.unknowns['blade%s:surface_norm' % self._suffix] = surfnorot

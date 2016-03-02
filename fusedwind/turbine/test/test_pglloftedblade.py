
import numpy as np
import unittest
import os
import pkg_resources

from openmdao.api import Problem, Group

from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       PGLRedistributedPlanform

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')

def configure(cfg):

    pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))
    nsec = 8
    s_new = np.linspace(0, 1, nsec)
    pf = redistribute_planform(pf, s=s_new)

    cfg['redistribute_flag'] = False
    cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])
    afs = []
    for f in [os.path.join(PATH, 'data/ffaw3241.dat'),
              os.path.join(PATH, 'data/ffaw3301.dat'),
              os.path.join(PATH, 'data/ffaw3360.dat'),
              os.path.join(PATH, 'data/cylinder.dat')]:

        afs.append(np.loadtxt(f))
    cfg['base_airfoils'] = afs
    d = PGLLoftedBladeSurface(cfg, size_in=nsec, size_out=(200, nsec, 3), suffix='_st')
    p = Problem(root=Group())
    r = p.root.add('blade_surf', d, promotes=['*'])
    p.setup()
    for k, v in pf.iteritems():
        if 'blade_st:'+k in p.root.blade_surf.params.keys():
            p.root.blade_surf.params['blade_st:' + k] = v

    return p

class PGLLoftedBladeSurfaceTestCase(unittest.TestCase):

    def test_surf(self):

        cfg = {}

        p = configure(cfg)
        p.run()
        self.assertAlmostEqual(np.sum(p['blade_st:surface']), 775.21809362184081, places=6)

if __name__ == '__main__':

    unittest.main()
    # p = configure({})
    # p.run()

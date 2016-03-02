
import numpy as np
import unittest

from openmdao.api import Problem, Group

from fusedwind.turbine.geometry import PGLRedistributedPlanform


def configure(size_in=10, size_out=20):

    pf = {name:np.linspace(0, 1, 10) for name in ['s', 'x', 'y', 'z',
                                            'rot_x', 'rot_y', 'rot_z', 'chord', 'rthick','p_le']}
    s_new = np.linspace(0, 1, size_out)

    p = Problem(root=Group())
    r = p.root.add('redist', PGLRedistributedPlanform('_st', size_in, s_new), promotes=['*'])
    p.setup()
    for k, v in pf.iteritems():
        r.params['blade_ae:' + k] = v

    return p

class PlanformTestCase(unittest.TestCase):

    def test_redist(self):
        size_in = 10
        size_out = 20
        p = configure(size_in, size_out)
        p.run()
        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_st:x'], np.linspace(0, 1, size_out), decimal=4), None)

if __name__ == '__main__':

    unittest.main()
    # p = configure()
    # p.run()

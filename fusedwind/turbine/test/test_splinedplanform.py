

import unittest
import os
import pkg_resources
import numpy as np
from openmdao.api import Problem, Group

from fusedwind.turbine.geometry import SplinedBladePlanform, \
                                       read_blade_planform, \
                                       redistribute_planform

chord_bez = np.array([ 0.06229255,  0.06265925,  0.06494163,  0.069981  ,  0.07472148,
        0.07708992,  0.0774626 ,  0.07638454,  0.07427456,  0.07151453,
        0.0682293 ,  0.06454157,  0.06055236,  0.056303  ,  0.05177262,
        0.04691412,  0.04162634,  0.03567898,  0.02759722,  0.00694712])
chord_pchip = np.array([ 0.06229255,  0.06229298,  0.06369571,  0.06755303,  0.07093679,
        0.07192255,  0.07219044,  0.0725103 ,  0.07292919,  0.07345478,
        0.07383831,  0.07383212,  0.07317304,  0.07154825,  0.06859791,
        0.0625421 ,  0.05146539,  0.0385012 ,  0.02584383,  0.00694712])
athick_pchip = np.array([ 0.06229255,  0.06055473,  0.05242633,  0.04247846,  0.03354131,
        0.02797533,  0.02462471,  0.02244562,  0.0209728 ,  0.01986875,
        0.01903185,  0.01837563,  0.01781751,  0.0172574 ,  0.01653171,
        0.01507261,  0.01240316,  0.00927879,  0.00622836,  0.00167426])

PATH = pkg_resources.resource_filename('fusedwind', 'turbine/test')

def configure(spline_type):

    pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))
    pf = redistribute_planform(pf, s=np.linspace(0, 1, 20))

    p = Problem(root=Group())
    spl = p.root.add('pf_splines', SplinedBladePlanform(pf), promotes=['*'])
    for name in ['x', 'chord', 'rot_z', 'rthick']:
        spl.add_spline(name, np.array([0, 0.25, 0.75, 1.]), spline_type=spline_type)
    spl.configure()
    p.setup()
    return p

class TestSplinedPlanform(unittest.TestCase):


    def test_bezier(self):
        p = configure('bezier')
        p['blade_ae:chord_C'][2] = 0.03
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_ae:chord'], chord_bez, decimal=6), None)

    def test_pchip(self):
        p = configure('pchip')
        p['blade_ae:chord_C'][2] = 0.03
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_ae:chord'], chord_pchip, decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_ae:athick'], athick_pchip, decimal=6), None)
        self.assertAlmostEqual(p['blade_curve_length'], 1.0011587264848194, places=6)

    def test_curve_length(self):
        p = configure('pchip')
        p['blade_ae:x_C'][3] = 0.03
        p.run()
        self.assertAlmostEqual(p['blade_curve_length'], 1.0032258904912261, places=6)

    def test_bladescale(self):
        """
        test for
        """
        chord = np.array([ 0.05662959,  0.05662998,  0.05790519,  0.06141184,  0.06448799,
        0.06532847,  0.06433541,  0.06205359,  0.05890751,  0.05528587,
        0.05134422,  0.04723913,  0.04311308,  0.0390634 ,  0.03514467,
        0.0314088 ,  0.02789184,  0.02456755,  0.02037705,  0.00631557])

        p = configure('pchip')
        p['blade_scale'] = 1.1
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_ae:chord'], chord, decimal=6), None)

if __name__ == '__main__':

    unittest.main()
    # p = configure('pchip')
    # p['chord_C'][2] = 0.03
    #
    # p.run()

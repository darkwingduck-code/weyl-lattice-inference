"""Meaningful small all-band/normalization and fit controls for boundary refinement."""
import unittest
import numpy as np
import detector_boundary_refinement as target
import parameter_sector_sampling as sampler
from lattice_response import Parameters,interaction_matrix,rpa_response
from magnetic_bloch_response import magnetic_bloch_spectrum

class BoundaryRefinementTests(unittest.TestCase):
    def test_tiled_all_band_q_cases_match_direct_small_twisted_torus(self):
        p=Parameters(Lx=3,Ly=3,Nkz=48,theta_x=.21,theta_y=.37,theta_z=.19)
        specs=sampler.case_specs(np.array([.05,.2,.3]),p.Nkz,('baseline','qhalf','qdouble'))
        result,completed=target.accumulate(p,0,1,2,specs)
        self.assertEqual(completed,9)
        spectrum=magnetic_bloch_spectrum(p,0,1)
        for name,s in specs.items():
            v=sampler.transition_vertices(spectrum,s['qshift'])
            tr,stats=sampler.weighted_transitions(spectrum,v,.5,.04)
            direct=sampler.two_sector_response(tr,s['omega'],s['eta'])
            self.assertEqual(result[name]['raw_pairs'],9*48*16)
            self.assertEqual(stats['raw_pairs'],9*48*16)
            self.assertLess(np.max(abs(result[name]['bare']-direct)),3e-13)
            self.assertLess(np.max(abs(result[name]['current_probe']-tr.full_response(.275*s['q']/sampler.BASE_Q+1j*s['eta']))),3e-13)
            self.assertLess(abs(result[name]['density']-tr.density_value),2e-14)
            self.assertLess(np.max(abs(result[name]['moment0'][0])),2e-14)
            u=interaction_matrix(v.Q,.8,8.)
            got=np.array([rpa_response(x,u) for x in result[name]['bare'][0]])
            expected=np.array([rpa_response(x,u) for x in direct[0]])
            self.assertLess(np.max(abs(got-expected)),3e-13)

    def test_nonlinear_rpa_must_follow_total_bare_sum(self):
        p=np.array([[.05-.02j,.01+.005j],[.009-.006j,-.2-.01j]])
        u=np.diag([.8,8.])
        correct=rpa_response(2*p,u)
        wrong=2*rpa_response(p,u)
        self.assertGreater(np.max(abs(correct-wrong)),.01)

    def test_profile_restores_known_coulomb_parameter(self):
        pi=np.array([[[.12-.04j,.02+.01j],[.03-.02j,-.47-.03j]],
                     [[.21-.06j,.01-.03j],[.02+.04j,-.36-.02j]]])
        data={'a':dict(pi=pi,q=.3)}
        truth=target.truths_for(data)
        fit=target.profile(data,truth,('a',),8.)
        self.assertLess(abs(fit['g']-.8),2e-8)
        self.assertLess(fit['normalized_mismatch'],2e-8)
        self.assertTrue(fit['optimizer_comparison_pass'])

    def test_shared_reference_exposes_input_error_own_truth_cancels(self):
        pi=np.array([[[.12-.04j,.02+.01j],[.03-.02j,-.47-.03j]],
                     [[.21-.06j,.01-.03j],[.02+.04j,-.36-.02j]]])
        fine={'a':dict(pi=pi,q=.3)}
        coarse_pi=pi.copy();coarse_pi[0,0,0]+=.025j
        coarse={'a':dict(pi=coarse_pi,q=.3)}
        own=target.profile(coarse,target.truths_for(coarse),('a',),8.)
        shared=target.profile(coarse,target.truths_for(fine),('a',),8.)
        self.assertLess(own['normalized_mismatch'],2e-8)
        self.assertGreater(shared['normalized_mismatch'],1e-4)
        self.assertTrue(shared['optimizer_comparison_pass'])

if __name__=='__main__':unittest.main()
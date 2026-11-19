import numpy as np
from nmnoise.core import Pulse, X, Y, Z, ghz_schedule, ideal_state, ensemble_rho, trace_distance
from nmnoise.classical import ou_noise, dense_trajectories, lindblad, ou_chi
from nmnoise.mps import mps_trajectories


def assert_density(rho, tolerance=1e-9):
    np.testing.assert_allclose(rho, rho.conj().T, atol=tolerance)
    np.testing.assert_allclose(np.trace(rho), 1, atol=tolerance)
    assert np.linalg.eigvalsh(rho).min() > -tolerance


def test_zero_noise_produces_ghz_in_both_representations():
    p = ghz_schedule(3)
    target = np.zeros(8, complex)
    target[[0,7]] = 1/np.sqrt(2)
    np.testing.assert_allclose(ideal_state(p,3), target, atol=1e-12)
    noise = ou_noise(p,3,4,samples=1,sigma=0)
    a = dense_trajectories(p,noise,4)
    b, diagnostics = mps_trajectories(p,noise,4)
    assert abs(np.vdot(target,a[0]))**2 > 1-1e-12
    assert abs(np.vdot(target,b[0]))**2 > 1-1e-12
    assert diagnostics['peak_bond'] == 2


def test_mps_matches_unsplit_dense_and_strang_converges():
    # Sweep the active bond in both directions; random initial entanglement
    # exercises canonicalization and index ordering, not only product inputs.
    p = ghz_schedule(3)
    p.append(Pulse('XX reverse bond',20.,(0,1),0.03*np.kron(X,X)))
    rng=np.random.default_rng(41)
    initial=rng.normal(size=8)+1j*rng.normal(size=8)
    initial/=np.linalg.norm(initial)
    noise=ou_noise(p,3,4,samples=3,sigma=.02)
    a=ensemble_rho(dense_trajectories(p,noise,4,initial))
    b,_=mps_trajectories(p,noise,4,initial,substeps=1)
    c,_=mps_trajectories(p,noise,4,initial,substeps=2)
    e1=trace_distance(a,ensemble_rho(b))
    e2=trace_distance(a,ensemble_rho(c))
    assert e2 < .3*e1
    assert e2 < 1e-3
    assert_density(a)


def test_bond_truncation_is_reported():
    p=ghz_schedule(3)
    noise=ou_noise(p,3,4,samples=1,sigma=0)
    states,info=mps_trajectories(p,noise,4,max_bond=1)
    assert info['max_trajectory_discarded_weight_sum'] > 0
    assert info['peak_bond'] == 1
    assert abs(np.vdot(ideal_state(p,3), states[0]))**2 < .6


def test_ou_ramsey_matches_analytic_continuous_covariance():
    p=[Pulse('idle',80.,(0,),np.zeros((2,2),complex))]
    plus=np.ones(2)/np.sqrt(2)
    noise=ou_noise(p,1,1,samples=8192,sigma=.008)
    states=dense_trajectories(p,noise,1,plus)
    # X expectation samples; finite sampling tolerance determined from variance.
    xs=2*(states[:,0]*states[:,1].conj()).real
    expected=np.exp(-ou_chi(80,sigma=.008))
    se=xs.std(ddof=1)/np.sqrt(len(xs))
    assert abs(xs.mean()-expected) < 5*se+1e-4
    assert_density(ensemble_rho(states))


def test_lindblad_ramsey_rate_convention():
    p=[Pulse('idle',80.,(0,),np.zeros((2,2),complex))]
    rho=lindblad(p,1,.01,np.ones(2)/np.sqrt(2))
    np.testing.assert_allclose(2*rho[0,1],np.exp(-.8),atol=1e-12)
    assert_density(rho)


def test_quantum_heom_against_independent_explicit_bath():
    from nmnoise.quantum import heom, explicit_bath
    a,_=heom(depth=6)
    b,_=explicit_bath(fock=7)
    assert trace_distance(a[-1],b[-1]) < 2e-5
    for rho in a:
        assert_density(rho,tolerance=1e-7)


def test_heom_pure_dephasing_analytic_and_memory_continuity():
    from nmnoise.quantum import heom, mode_parameters
    plus=np.ones((2,2))/2
    p=[Pulse('idle 1',15.,(0,),np.zeros((2,2),complex)),
       Pulse('idle 2',15.,(0,),np.zeros((2,2),complex))]
    a,_=heom(p,depth=6,initial=plus)
    r,_=heom(p,depth=6,initial=plus,reset=True)
    w,g,nb=mode_parameters()
    exponent=np.sum(g*g*(2*nb+1)*(1-np.cos(w*30))/w**2)
    np.testing.assert_allclose(2*a[-1,0,1],np.exp(-exponent),atol=2e-5)
    # A merely artificial boundary cannot change the physical evolution.
    whole=[Pulse('idle',30.,(0,),np.zeros((2,2),complex))]
    b,_=heom(whole,depth=6,initial=plus)
    assert trace_distance(a[-1],b[-1]) < 1e-7
    assert trace_distance(a[-1],r[-1]) > 1e-3

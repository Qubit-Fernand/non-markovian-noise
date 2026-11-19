import numpy as np
import pytest
from scipy.integrate import quad
from nmnoise.core import X, Y, Z, ideal_state, Pulse
from nmnoise.classical import dense_trajectories, lindblad
from nmnoise.paired import (paired_schedules, pulse_schedule, shared_grid,
                           integrated_ou_noise, propagate_bloch, instantaneous_coherence,
                           fit_lindblad_rate, ou_interval_coefficients)


def test_paired_identity_timing_and_invalid_overlap():
    for schedule in paired_schedules(200, 20):
        assert sum(p.duration for p in schedule) == 200
        for psi in (np.array([1., 0.]), np.ones(2)/np.sqrt(2)):
            np.testing.assert_allclose(ideal_state(schedule, 1, psi), -psi, atol=1e-12)
    with pytest.raises(ValueError):
        paired_schedules(100, 40)
    with pytest.raises(ValueError):
        pulse_schedule(100, 20, [60, 50])


def test_fast_bloch_against_existing_dense_solver():
    schedules = paired_schedules(200, 20)
    grid = np.arange(201.)
    noise = np.random.default_rng(4).normal(scale=.03, size=(12, 200))
    bloch = propagate_bloch(schedules, grid, noise)
    for k, schedule in enumerate(schedules):
        states = dense_trajectories(schedule, noise[:, :, None], 1, np.ones(2)/np.sqrt(2))
        expected = np.array([np.einsum('bi,ij,bj->b', states.conj(), op, states).real
                             for op in (X, Y, Z)]).T
        np.testing.assert_allclose(bloch[k], expected, atol=1e-12)


def test_joint_ou_endpoint_integral_covariance_and_small_step():
    # Independent double-integral check, including tau much longer than a cell.
    for tau in (1., 100., 1e6):
        h = .5
        a, b, l11, l21, l22 = [float(x[0]) for x in ou_interval_coefficients(h, [tau])]
        exact_var = 2*quad(lambda t: (h-t)*np.exp(-t/tau), 0, h, epsabs=1e-12)[0]
        np.testing.assert_allclose(b*b+l21*l21+l22*l22, exact_var, rtol=1e-11)
        np.testing.assert_allclose(a*b+l11*l21, tau*(-np.expm1(-h/tau)), rtol=1e-11)
        assert l22 > 0


def test_integrated_ou_pulse_filter_matches_exact_gaussian_result():
    times = np.array([0., 25., 125., 200.])
    taus, sigma = [30., 300.], .015
    noise = integrated_ou_noise(times, taus, 32768, sigma, 61)
    phase = noise @ (np.diff(times)*np.array([1., -1., 1.]))
    xs = np.cos(phase)
    exact = instantaneous_coherence(200, [25, 125], taus, sigma)
    assert abs(xs.mean()-exact) < 5*xs.std(ddof=1)/np.sqrt(len(xs))
    # Independent direct quadrature of the covariance with switching points.
    durations = np.diff(times)
    chi = 0.
    for i in range(3):
        for j in range(3):
            value = quad(lambda t: quad(lambda s: np.mean(np.exp(-abs(t-s)/np.array(taus))),
                         times[j], times[j+1], points=[t] if times[j]<t<times[j+1] else None)[0],
                         times[i], times[i+1], epsabs=1e-7)[0]
            chi += .5*sigma**2*(-1)**(i+j)*value
    np.testing.assert_allclose(exact, np.exp(-chi), atol=1e-8)


def test_lindblad_pair_is_identical_and_fit_recovers_rate():
    gamma = .002
    for schedule in paired_schedules(200, 20):
        rho = lindblad(schedule, 1, gamma, np.ones(2)/np.sqrt(2))
        np.testing.assert_allclose(np.trace(rho@X).real, np.exp(-gamma*200), atol=1e-12)
    times = np.array([100., 400., 1600.]*2)
    fitted = fit_lindblad_rate(times, np.exp(-gamma*times))
    np.testing.assert_allclose(fitted['gamma_phi_per_ns'], gamma, rtol=1e-6)


def test_finite_pulse_time_ordering_convergence():
    schedules = paired_schedules(200, 20)
    results = []
    for step in (4., 2., .125):
        times = shared_grid(schedules, step)
        # Exact averages of a smooth detuning, independent of OU sampling.
        noise = (.03*25*(np.cos(times[:-1]/25)-np.cos(times[1:]/25))/np.diff(times))[None, :]
        results.append(propagate_bloch(schedules, times, noise))
    e1 = np.linalg.norm(results[0]-results[2])
    e2 = np.linalg.norm(results[1]-results[2])
    assert e2 < .35*e1
    assert e2 < .002


def test_zero_noise_and_quasistatic_ideal_echo_limits():
    schedules = paired_schedules(200, 20)
    times = shared_grid(schedules, 1)
    actual = propagate_bloch(schedules, times, np.zeros((2, len(times)-1)))
    np.testing.assert_allclose(actual, np.broadcast_to([1., 0., 0.], actual.shape), atol=1e-12)
    for centers in ([50, 150], [25, 125]):
        assert instantaneous_coherence(200, centers, [1e10], .01) > 1-1e-7

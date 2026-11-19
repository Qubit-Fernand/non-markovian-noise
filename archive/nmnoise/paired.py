"""Single-qubit paired-pulse experiments, in ns and rad/ns.

OU endpoints AND interval integrals are sampled jointly exactly. Idle evolution
is therefore exact for the sampled integrated noise. During a finite pulse the
interval-average noise is used in the full Hamiltonian exponential; pulse_dt
controls this time-ordering approximation. This is classical Gaussian noise,
not a quantum bath or a general non-Markovianity witness.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from .core import Pulse, X, Y, Z


def pulse_schedule(total_ns, width_ns, centers):
    """Rectangular X-pi pulses with specified centers; total includes pulses."""
    centers = np.asarray(centers, dtype=float)
    if (not np.isfinite(total_ns) or not np.isfinite(width_ns)
            or total_ns <= 0 or width_ns <= 0 or not np.all(np.isfinite(centers))):
        raise ValueError("Times must be finite and positive")
    if np.any(np.diff(centers) < width_ns):
        raise ValueError("Pulses overlap or centers are not ordered")
    pulses, cursor = [], 0.
    for center in centers:
        start, end = center-width_ns/2, center+width_ns/2
        if start < cursor-1e-10 or end > total_ns+1e-10:
            raise ValueError("Pulse does not fit inside the experiment")
        if start > cursor+1e-10:
            pulses.append(Pulse("idle", start-cursor, (0,), np.zeros((2, 2), complex)))
        pulses.append(Pulse("X(pi)", width_ns, (0,), np.pi/(2*width_ns)*X))
        cursor = end
    if total_ns > cursor+1e-10:
        pulses.append(Pulse("idle", total_ns-cursor, (0,), np.zeros((2, 2), complex)))
    return pulses


def paired_schedules(total_ns, width_ns=20.):
    """A centers: T/4,3T/4. B centers: T/8,5T/8. Initial state is |+>."""
    return [pulse_schedule(total_ns, width_ns, total_ns*np.array(fractions))
            for fractions in ((.25, .75), (.125, .625))]


def shared_grid(schedules, pulse_dt=1.):
    """Union of all control boundaries; refine driven segments only."""
    if not np.isfinite(pulse_dt) or pulse_dt <= 0:
        raise ValueError("pulse_dt must be finite and positive")
    times = {0.}
    totals = []
    for schedule in schedules:
        cursor = 0.
        for pulse in schedule:
            if pulse.duration <= 0:
                raise ValueError("Pulse durations must be positive")
            steps = max(1, math.ceil(pulse.duration/pulse_dt)) if np.any(pulse.h) else 1
            times.update(np.round(np.linspace(cursor, cursor+pulse.duration, steps+1), 12))
            cursor += pulse.duration
        totals.append(cursor)
    if not totals or not np.allclose(totals, totals[0], rtol=0, atol=1e-9):
        raise ValueError("All schedules must have the same total duration")
    return np.array(sorted(times))


def ou_interval_coefficients(duration, taus):
    """Unit-variance OU: x'=a*x+l11*z1, integral=b*x+l21*z1+l22*z2."""
    taus = np.asarray(taus, dtype=float)
    if (duration <= 0 or not np.isfinite(duration) or taus.ndim != 1
            or len(taus) == 0 or np.any(taus <= 0) or not np.all(np.isfinite(taus))):
        raise ValueError("Positive finite duration and correlation times required")
    u = duration/taus
    a = np.exp(-u)
    one_minus_a = -np.expm1(-u)
    endpoint_var = -np.expm1(-2*u)
    # 2u-3+4e^-u-e^-2u; Taylor series prevents cancellation for slow modes.
    integrated_var = 2*u-2*one_minus_a-one_minus_a**2
    small = u < .01
    integrated_var[small] = sum(
        (4*(-1)**n-(-2)**n)/math.factorial(n)*u[small]**n for n in range(3, 11))
    integrated_var *= taus**2
    cross_cov = taus*one_minus_a**2
    l11 = np.sqrt(endpoint_var)
    l21 = cross_cov/l11
    l22 = np.sqrt(np.maximum(0, integrated_var-l21**2))
    return a, taus*one_minus_a, l11, l21, l22


def integrated_ou_noise(times, taus, samples, sigma, seed):
    """Interval-average OU-mixture noise, shape (samples, intervals).

    All independent modes have variance sigma**2/len(taus); initial conditions
    are stationary. The same returned array can drive several pulse sequences.
    """
    times = np.asarray(times, dtype=float)
    if (samples < 2 or sigma < 0 or not np.isfinite(sigma) or times.ndim != 1
            or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid sample count, noise strength, or grid")
    taus = np.asarray(taus, dtype=float)
    ou_interval_coefficients(times[1]-times[0], taus)  # validate before RNG allocation
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(samples, len(taus)))
    noise = np.empty((samples, len(times)-1))
    scale = sigma/np.sqrt(len(taus))
    for j, duration in enumerate(np.diff(times)):
        a, b, l11, l21, l22 = ou_interval_coefficients(duration, taus)
        z1, z2 = rng.normal(size=(2, samples, len(taus)))
        integral = b*state+l21*z1+l22*z2
        noise[:, j] = scale*integral.sum(axis=1)/duration
        state = a*state+l11*z1
    return noise


def propagate_bloch(schedules, times, noise, initial=(1., 0., 0.)):
    """Exact cellwise SU(2) rotation for H_control + xi*Z/2; no splitting.

    Returns per-trajectory Bloch vectors, shape (sequences,samples,3).
    Equivalent to dense_trajectories on the same piecewise-constant grid.
    """
    times, noise = np.asarray(times), np.asarray(noise)
    if noise.ndim != 2 or noise.shape[1] != len(times)-1 or np.any(np.diff(times) <= 0):
        raise ValueError("Noise/grid mismatch")
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)) or np.linalg.norm(initial) > 1+1e-12:
        raise ValueError("Initial Bloch vector must represent a density matrix")
    controls = np.empty((len(schedules), len(times)-1, 3))
    for k, schedule in enumerate(schedules):
        bounds = np.r_[0., np.cumsum([p.duration for p in schedule])]
        if not np.isclose(bounds[-1], times[-1]) or not np.isclose(times[0], 0):
            raise ValueError("Schedule/grid duration mismatch")
        if any(not np.any(np.isclose(times, b, rtol=0, atol=1e-9)) for b in bounds):
            raise ValueError("Grid must contain every control boundary")
        ids = np.searchsorted(bounds[1:], (times[1:]+times[:-1])/2)
        vectors = np.array([[np.trace(p.h @ op).real for op in (X, Y, Z)] for p in schedule])
        controls[k] = vectors[ids]
    r = np.broadcast_to(initial, (len(schedules), len(noise), 3)).copy()
    for j, duration in enumerate(np.diff(times)):
        omega = np.broadcast_to(controls[:, j, None, :], r.shape).copy()
        omega[:, :, 2] += noise[None, :, j]
        angle = np.linalg.norm(omega, axis=2)*duration
        # Rodrigues formula, written with sinc to include zero angular speed.
        cross = np.cross(omega, r)
        r = (r + (duration*np.sinc(angle/np.pi))[..., None]*cross
             + (.5*duration**2*np.sinc(angle/(2*np.pi))**2)[..., None]*np.cross(omega, cross))
    return r


def instantaneous_coherence(total_ns, centers, taus, sigma):
    """Exact Gaussian dephasing coherence for instantaneous pi pulses.

    Integrates exp(-abs(t-s)/tau) on the signed idle intervals analytically.
    This is a reference, not the finite-pulse simulation.
    """
    edges = np.r_[0., centers, total_ns]
    taus = np.asarray(taus, dtype=float)
    if np.any(np.diff(edges) < 0) or np.any(taus <= 0) or sigma < 0:
        raise ValueError("Invalid ideal-pulse parameters")
    history, chi = np.zeros_like(taus), np.zeros_like(taus)
    for j, duration in enumerate(np.diff(edges)):
        y = (-1.)**j
        u = duration/taus
        one_minus_a = -np.expm1(-u)
        residual = u+np.expm1(-u)
        small = u < .01
        residual[small] = sum((-u[small])**n/math.factorial(n) for n in range(2, 11))
        chi += y*history*taus*one_minus_a+taus**2*residual
        history = np.exp(-u)*history+y*taus*one_minus_a
    return float(np.exp(-sigma**2*chi.mean()))


def fit_lindblad_rate(durations, x_expectations):
    """Shared gamma fitted to synthetic calibration X probabilities.

    Equal weights per sequence, expected binomial negative log likelihood.
    No test-sequence data enter this fit; no real shot noise or nuisance fit.
    For |+>, rectangular X pulses and Z dephasing, X(T)=exp(-gamma*T).
    """
    durations, xs = np.asarray(durations), np.asarray(x_expectations)
    if durations.shape != xs.shape or np.any(durations <= 0) or np.any(np.abs(xs) > 1+1e-10):
        raise ValueError("Invalid calibration arrays")
    truth = (1+np.clip(xs, -1, 1))/2

    def objective(log_rate):
        p = np.clip((1+np.exp(-np.exp(log_rate)*durations))/2, 1e-12, 1-1e-12)
        return float(-np.mean(truth*np.log(p)+(1-truth)*np.log1p(-p)))

    result = minimize_scalar(objective, bounds=(-30., 0.), method="bounded",
                             options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError("Lindblad calibration failed")
    gamma = float(np.exp(result.x))
    return dict(gamma_phi_per_ns=gamma, expected_binomial_nll=float(result.fun),
                x_predictions=np.exp(-gamma*durations).tolist(),
                rmse=float(np.sqrt(np.mean((np.exp(-gamma*durations)-xs)**2))))


def summarize_pair(bloch):
    """MC errors use common noise; hardware shot errors must use independent shots."""
    if bloch.shape[0] != 2 or bloch.shape[1] < 2:
        raise ValueError("Two sequences and at least two trajectories required")
    means = bloch.mean(axis=1)
    errors = bloch.std(axis=1, ddof=1)/np.sqrt(bloch.shape[1])
    difference = bloch[0, :, 0]-bloch[1, :, 0]
    delta = float(difference.mean())
    # Conservative independent-shot bound at 3 standard deviations, ignoring
    # SPAM, drift, model uncertainty, multiple testing, and power requirements.
    shots = math.ceil(18/delta**2) if abs(delta) > 1e-12 else None
    return dict(bloch_mean=means.tolist(), bloch_mc_se=errors.tolist(),
                delta_x=delta, delta_x_mc_se=float(difference.std(ddof=1)/np.sqrt(len(difference))),
                shots_per_sequence_3se_worst_case=shots)

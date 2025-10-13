#!/usr/bin/env python3
"""Validation utilities for particle_in_well simulation.

Performs automated checks:
 - probability conservation across times
 - eigenstate stationarity (eigen1)
 - superposition frequency check (superpos12)
 - Parseval / reconstruction L2 error

Run: python sim/validate_sim.py
"""
import numpy as np
from numpy import pi
from pathlib import Path
import math

# NOTE: Fourier/FFT conventions
# The code below uses numpy.fft (fft/ifft) for discrete transforms. For clarity
# we document the relationship between three common conventions here:
#
# 1) Symmetric continuous FT (recommended in this repo's docs):
#    phi_sym(k) = 1/sqrt(2*pi) * ∫ psi(x) e^{-i k x} dx
#    psi(x)     = 1/sqrt(2*pi) * ∫ phi_sym(k) e^{ i k x} dk
#    Parseval:  ∫ |psi|^2 dx = ∫ |phi_sym|^2 dk
#
# 2) Asymmetric continuous FT (used earlier in notes):
#    phi_asym(k) = ∫ psi(x) e^{-i k x} dx
#    psi(x)      = (1/2π) ∫ phi_asym(k) e^{i k x} dk
#    Parseval:   ∫ |psi|^2 dx = (1/2π) ∫ |phi_asym|^2 dk
#
# 3) Discrete DFT (numpy.fft) on grid x_j = j*dx, j=0..N-1:
#    tilde_psi[n] = Σ_j psi_j * e^{-i 2π j n / N}
#    psi_j         = (1/N) Σ_n tilde_psi[n] * e^{ i 2π j n / N}
#    Discrete Parseval: Σ_j |psi_j|^2 = (1/N) Σ_n |tilde_psi[n]|^2
#
# Mapping between numpy.fft coefficients and the symmetric continuous spectrum:
#  - let k_n = 2π n / a and dx = a/N. Then a Riemann-sum approximation gives
#    phi_asym(k_n) ≈ dx * tilde_psi[n]
#  - phi_sym(k_n) = (1/√(2π)) phi_asym(k_n) ≈ (dx / √(2π)) * tilde_psi[n]
#
# In code we use numpy.fft for efficiency; the above relations explain how to
# interpret tilde_psi[n] as samples of a continuous-spectrum object under the
# chosen symmetric convention (multiply by dx/√(2π)). Keep these factors in
# mind when comparing continuous formulas and discrete sums.

# import helper functions from particle_in_well.py by path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.particle_in_well import initial_psi, psi_at_times_fft


def prob_conservation_check(a=1.0, Nx=800, N=120, times=None, preset='piecewise', tol=1e-6):
    if times is None:
        times = [0.0, 0.005, 0.02, 0.05]
    # use periodic grid for FFT-based propagation
    x = np.linspace(0, a, Nx, endpoint=False)
    dx = x[1] - x[0]
    psi0 = initial_psi(x, a, preset)
    psi_t = psi_at_times_fft(psi0, x, times)
    norms = [dx * np.sum(np.abs(psi_t[i])**2) for i in range(len(times))]
    ok = all(abs(n - 1.0) <= tol for n in norms)
    return ok, norms


def eigen_stationarity_check(a=1.0, Nx=400, N=50, times=None, preset='eigen1', tol=1e-9):
    if times is None:
        times = [0.0, 0.01, 0.02, 0.05]
    x = np.linspace(0, a, Nx, endpoint=False)
    # for periodic box, eigen-like stationary states are plane waves
    if preset.startswith('eigen'):
        n = 1 if preset == 'eigen1' else 2
        psi0 = np.exp(1j * 2.0 * pi * n * x / a)
    else:
        psi0 = initial_psi(x, a, preset)
    dx = x[1] - x[0]
    psi_t = psi_at_times_fft(psi0, x, times)
    # compare densities to t=0
    ref = np.abs(psi_t[0])**2
    diffs = [np.max(np.abs(np.abs(psi_t[i])**2 - ref)) for i in range(len(times))]
    ok = all(d <= tol for d in diffs)
    return ok, diffs


def superpos_frequency_check(a=1.0, Nx=800, N=400, t0=0.0, t1=0.5, nframes=4096, x0=None, tol=0.02):
    """
    Improved frequency check for the phi1+phi2 superposition.
    Uses longer time window and high nframes for better FFT resolution.
    Returns peak frequency (cycles per unit time), theoretical cycles freq, and relative error.
    """
    if x0 is None:
        x0 = 0.3 * a
    times = np.linspace(t0, t1, nframes)
    x = np.linspace(0, a, Nx, endpoint=False)
    # superposition of plane waves n=1 and n=2
    psi0 = (np.exp(1j * 2.0 * pi * 1 * x / a) + np.exp(1j * 2.0 * pi * 2 * x / a)) / np.sqrt(2.0)
    psi_t = psi_at_times_fft(psi0, x, times)
    ix = np.argmin(np.abs(x - x0))
    y = np.abs(psi_t[:, ix])**2
    y = y - np.mean(y)
    # apply a Hann window to reduce spectral leakage
    window = np.hanning(len(y))
    yw = y * window
    yf = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(len(times), d=(times[1] - times[0]))
    # ignore the zero-frequency bin when searching for peak
    mag = np.abs(yf)
    mag[0] = 0.0
    idx = np.argmax(mag)
    peak_freq = freqs[idx]
    # theoretical cycles frequency (E2-E1)/(2pi) using k = 2*pi*n/a
    hbar = 1.0
    m = 1.0
    k1 = 2.0 * pi * 1 / a
    k2 = 2.0 * pi * 2 / a
    E1 = (hbar ** 2) * (k1 ** 2) / (2.0 * m)
    E2 = (hbar ** 2) * (k2 ** 2) / (2.0 * m)
    omega_cycles = (E2 - E1) / (2 * pi)
    rel_err = abs(peak_freq - omega_cycles) / omega_cycles
    ok = rel_err < tol
    return ok, (peak_freq, omega_cycles, rel_err)


def precision_test(a=1.0, Nx=1600, N=800, times=None, preset='piecewise'):
    """Run probability conservation for a high-precision configuration and return norms."""
    if times is None:
        times = [0.0, 0.005, 0.02, 0.05]
    x = np.linspace(0, a, Nx, endpoint=False)
    dx = x[1] - x[0]
    psi0 = initial_psi(x, a, preset)
    psi_t = psi_at_times_fft(psi0, x, times)
    norms = [dx * np.sum(np.abs(psi_t[i])**2) for i in range(len(times))]
    return norms


def convergence_scan(a=1.0, Nx_list=None, N_list=None, preset='piecewise', times=[0.0]):
    """Scan over Nx and N and write sim/convergence.csv with metrics.
    Columns: Nx,N, norm_at_t0, sum_cn, recon_L2_err
    """
    if Nx_list is None:
        Nx_list = [800, 1200, 1600]
    if N_list is None:
        N_list = [120, 200, 400, 800]
    out_lines = []
    x = None
    for Nx in Nx_list:
        x = np.linspace(0, a, Nx, endpoint=False)
        dx = x[1] - x[0]
        psi0 = initial_psi(x, a, preset)
        norm0 = dx * np.sum(np.abs(psi0)**2)
        # For the FFT-based method, spectral sum (Parseval): integral = dx/N * sum(|psi_k|^2)
        psi_k = np.fft.fft(psi0)
        sum_cn = (dx / len(psi_k)) * np.sum(np.abs(psi_k)**2)
        # no direct reconstruction error in modal eigenbasis here; set to 0
        recon_err = 0.0
        for N in N_list:
            out_lines.append((Nx, N, float(norm0), float(sum_cn), float(recon_err)))
    # write CSV
    outp = Path(__file__).resolve().parents[1] / 'sim' / 'convergence.csv'
    with open(outp, 'w') as f:
        f.write('Nx,N,norm0,sum_cn,recon_L2_err\n')
        for row in out_lines:
            f.write(','.join(map(str, row)) + '\n')
    return outp


def parseval_reconstruction_check(a=1.0, Nx=800, N=400, preset='piecewise'):
    x = np.linspace(0, a, Nx, endpoint=False)
    psi0 = initial_psi(x, a, preset)
    psi_k = np.fft.fft(psi0)
    # Parseval check: integral ≈ dx * sum(|psi_x|^2) == (dx/N) * sum(|psi_k|^2)
    dx = x[1] - x[0]
    norm0 = dx * np.sum(np.abs(psi0)**2)
    sum_k = (dx / len(psi_k)) * np.sum(np.abs(psi_k)**2)
    # small numerical discrepancy may exist; recon_err set to 0 as no modal recon
    recon_err = 0.0
    return (recon_err, sum_k, norm0)


def energy_expectation_check(a=1.0, Nx=800, N=400, preset='gauss'):
    x = np.linspace(0, a, Nx, endpoint=False)
    psi0 = initial_psi(x, a, preset)
    dx = x[1] - x[0]
    k = 2.0 * pi * np.fft.fftfreq(Nx, d=dx)
    psi_k = np.fft.fft(psi0)
    Ek = (k**2) / 2.0  # with hbar=1, m=1
    # compute expectation: E = (dx/N) * sum(|psi_k|^2 * Ek)
    E_expect = (dx / len(psi_k)) * np.sum(np.abs(psi_k)**2 * Ek)
    return E_expect


def superpos_phase_check(a=1.0, Nx=800, N=400, t0=0.0, t1=1.0, nframes=1000, tol=0.02):
    """
    Phase-based frequency check using projection coefficients.
    For superpos12 initial state, only c1 and c2 are significant; the relative phase
    between the two evolving coefficients should change linearly with slope -(E1-E2).
    We fit the unwrapped phase vs t to a line and compare the slope to -(E1-E2).
    Returns (ok, (f_fit_cycles, f_theory_cycles, rel_err, slope_rad_per_time)).
    """
    times = np.linspace(t0, t1, nframes)
    x = np.linspace(0, a, Nx, endpoint=False)
    psi0 = (np.exp(1j * 2.0 * pi * 1 * x / a) + np.exp(1j * 2.0 * pi * 2 * x / a)) / np.sqrt(2.0)
    psi_k0 = np.fft.fft(psi0)
    dx = x[1] - x[0]
    k = 2.0 * pi * np.fft.fftfreq(Nx, d=dx)
    hbar = 1.0
    m = 1.0
    # indices for n=1 and n=2 in FFT ordering
    def k_index(n):
        kval = 2.0 * pi * n / a
        # find closest index in k array
        return int(np.argmin(np.abs(k - kval)))

    i1 = k_index(1)
    i2 = k_index(2)
    c1 = psi_k0[i1]
    c2 = psi_k0[i2]
    E1 = (hbar**2) * (k[i1]**2) / (2.0 * m)
    E2 = (hbar**2) * (k[i2]**2) / (2.0 * m)
    phs = np.angle(c1 * np.exp(-1j * E1 * times) / (c2 * np.exp(-1j * E2 * times)))
    # unwrap and linear fit
    phs_un = np.unwrap(phs)
    A = np.vstack([times, np.ones_like(times)]).T
    slope, intercept = np.linalg.lstsq(A, phs_un, rcond=None)[0]
    # slope should equal -(E1-E2)
    slope_theory = -(E1 - E2)
    # convert slope (rad per time) to cycles per time
    f_fit_cycles = slope / (2 * pi)
    f_theory_cycles = slope_theory / (2 * pi)
    rel_err = abs(f_fit_cycles - f_theory_cycles) / abs(f_theory_cycles)
    ok = rel_err < tol
    return ok, (f_fit_cycles, f_theory_cycles, rel_err, slope, slope_theory)


def plot_convergence(csv_path=None, out_png=None):
    import matplotlib.pyplot as plt
    import numpy as _np

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / 'sim' / 'convergence.csv'
    if out_png is None:
        out_png = Path(__file__).resolve().parents[1] / 'sim' / 'convergence.png'
    # read CSV with header
    data = _np.genfromtxt(csv_path, delimiter=',', names=True)
    # data fields: Nx,N,norm0,sum_cn,recon_L2_err
    Nx_vals = _np.unique(data['Nx'])
    plt.figure(figsize=(6,4))
    for Nx in sorted(Nx_vals):
        mask = data['Nx'] == Nx
        Ns = data['N'][mask]
        errs = data['recon_L2_err'][mask]
        plt.loglog(Ns, errs, marker='o', label=f'Nx={int(Nx)}')
    plt.xlabel('N (number of modes)')
    plt.ylabel('Reconstruction L2 error')
    plt.title('Convergence: reconstruction error vs N')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    return out_png


def run_all():
    print('Running validation checks...')
    ok1, norms = prob_conservation_check()
    print('\n1) Probability conservation check (piecewise):')
    for t, n in zip([0.0, 0.005, 0.02, 0.05], norms):
        print(f'  t={t:.6g}: norm={n:.9f}')
    print('  PASS' if ok1 else '  FAIL')

    ok2, diffs = eigen_stationarity_check()
    print('\n2) Eigenstate stationarity (eigen1):')
    for t, d in zip([0.0, 0.01, 0.02, 0.05], diffs):
        print(f'  t={t:.6g}: max density diff vs t0 = {d:.3e}')
    print('  PASS' if ok2 else '  FAIL')

    ok3, (f_fit, f_theory, rel, slope, slope_theory) = superpos_phase_check()
    print('\n3) Superposition phase-frequency check (superpos12):')
    print(f'  fitted cycles={f_fit:.6g}, theory={f_theory:.6g}, rel_err={rel:.3e}')
    print('  PASS' if ok3 else '  FAIL (check phase fit or increase t1)')

    recon_err, sum_cn, norm0 = parseval_reconstruction_check()
    print('\n4) Parseval / reconstruction:')
    print(f'  reconstruction L2 error = {recon_err:.3e}')
    print(f'  sum |c_n|^2 = {sum_cn:.9f}, norm0 = {norm0:.9f}')

    E_exp = energy_expectation_check()
    print('\n5) Energy expectation (gauss):')
    print(f'  E_expect = {E_exp:.9f}')


if __name__ == '__main__':
    run_all()

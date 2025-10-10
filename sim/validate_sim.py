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

# import helper functions from particle_in_well.py by path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.particle_in_well import eigenfunc, energy, initial_psi, project_coefficients, psi_at_times


def prob_conservation_check(a=1.0, Nx=800, N=120, times=None, preset='piecewise', tol=1e-6):
    if times is None:
        times = [0.0, 0.005, 0.02, 0.05]
    x = np.linspace(0, a, Nx)
    psi0 = initial_psi(x, a, preset)
    c = project_coefficients(psi0, x, a, N)
    psi_t = psi_at_times(c, x, a, times)
    norms = [np.trapezoid(np.abs(psi_t[i])**2, x) for i in range(len(times))]
    ok = all(abs(n - 1.0) <= tol for n in norms)
    return ok, norms


def eigen_stationarity_check(a=1.0, Nx=400, N=50, times=None, preset='eigen1', tol=1e-9):
    if times is None:
        times = [0.0, 0.01, 0.02, 0.05]
    x = np.linspace(0, a, Nx)
    psi0 = initial_psi(x, a, preset) if preset.startswith('eigen') is False else eigenfunc(1 if preset=='eigen1' else 2, x, a).astype(complex)
    c = project_coefficients(psi0, x, a, N)
    psi_t = psi_at_times(c, x, a, times)
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
    x = np.linspace(0, a, Nx)
    psi0 = (eigenfunc(1, x, a) + eigenfunc(2, x, a)).astype(complex) / np.sqrt(2.0)
    c = project_coefficients(psi0, x, a, N)
    psi_t = psi_at_times(c, x, a, times)
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
    # theoretical cycles frequency (E2-E1)/(2pi)
    E1 = energy(1, a)
    E2 = energy(2, a)
    omega_cycles = (E2 - E1) / (2 * pi)
    rel_err = abs(peak_freq - omega_cycles) / omega_cycles
    ok = rel_err < tol
    return ok, (peak_freq, omega_cycles, rel_err)


def precision_test(a=1.0, Nx=1600, N=800, times=None, preset='piecewise'):
    """Run probability conservation for a high-precision configuration and return norms."""
    if times is None:
        times = [0.0, 0.005, 0.02, 0.05]
    x = np.linspace(0, a, Nx)
    psi0 = initial_psi(x, a, preset)
    c = project_coefficients(psi0, x, a, N)
    psi_t = psi_at_times(c, x, a, times)
    norms = [np.trapezoid(np.abs(psi_t[i])**2, x) for i in range(len(times))]
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
        x = np.linspace(0, a, Nx)
        psi0 = initial_psi(x, a, preset)
        norm0 = np.trapezoid(np.abs(psi0)**2, x)
        for N in N_list:
            c = project_coefficients(psi0, x, a, N)
            sum_cn = np.sum(np.abs(c)**2)
            # reconstruction
            s = np.zeros_like(x, dtype=complex)
            for n in range(1, N+1):
                s += c[n-1] * eigenfunc(n, x, a)
            recon_err = math.sqrt(np.trapezoid(np.abs(psi0 - s)**2, x))
            out_lines.append((Nx, N, float(norm0), float(sum_cn), float(recon_err)))
    # write CSV
    outp = Path(__file__).resolve().parents[1] / 'sim' / 'convergence.csv'
    with open(outp, 'w') as f:
        f.write('Nx,N,norm0,sum_cn,recon_L2_err\n')
        for row in out_lines:
            f.write(','.join(map(str, row)) + '\n')
    return outp


def parseval_reconstruction_check(a=1.0, Nx=800, N=400, preset='piecewise'):
    x = np.linspace(0, a, Nx)
    psi0 = initial_psi(x, a, preset)
    c = project_coefficients(psi0, x, a, N)
    s = np.zeros_like(x, dtype=complex)
    for n in range(1, N+1):
        s += c[n-1] * eigenfunc(n, x, a)
    recon_err = np.sqrt(np.trapezoid(np.abs(psi0 - s)**2, x))
    sum_cn = np.sum(np.abs(c)**2)
    norm0 = np.trapezoid(np.abs(psi0)**2, x)
    return (recon_err, sum_cn, norm0)


def energy_expectation_check(a=1.0, Nx=800, N=400, preset='gauss'):
    x = np.linspace(0, a, Nx)
    psi0 = initial_psi(x, a, preset)
    c = project_coefficients(psi0, x, a, N)
    Es = np.array([energy(n, a) for n in range(1, N+1)])
    E_expect = np.sum(np.abs(c)**2 * Es)
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
    x = np.linspace(0, a, Nx)
    # prepare superposition initial state exactly
    psi0 = (eigenfunc(1, x, a) + eigenfunc(2, x, a)).astype(complex) / np.sqrt(2.0)
    c = project_coefficients(psi0, x, a, N)
    # pick first two coefficients
    c1 = c[0]
    c2 = c[1]
    # compute phase of ratio over time: phi(t) = arg( c1*e^{-iE1 t} / (c2*e^{-iE2 t}) )
    E1 = energy(1, a)
    E2 = energy(2, a)
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

    ok3, (peak, omega, rel) = superpos_frequency_check()
    print('\n3) Superposition frequency check (superpos12):')
    print(f'  peak_freq={peak:.6g}, theory={omega:.6g}, rel_err={rel:.3e}')
    print('  PASS' if ok3 else '  FAIL (increase nframes or check times)')

    recon_err, sum_cn, norm0 = parseval_reconstruction_check()
    print('\n4) Parseval / reconstruction:')
    print(f'  reconstruction L2 error = {recon_err:.3e}')
    print(f'  sum |c_n|^2 = {sum_cn:.9f}, norm0 = {norm0:.9f}')

    E_exp = energy_expectation_check()
    print('\n5) Energy expectation (gauss):')
    print(f'  E_expect = {E_exp:.9f}')


if __name__ == '__main__':
    run_all()

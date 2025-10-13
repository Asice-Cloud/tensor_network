#!/usr/bin/env python3
"""
Time evolution of a free particle on [0, a] using a spectral (FFT) method.

This code assumes periodic boundary conditions on [0,a]. The free-particle
time evolution is diagonal in k-space: psi_k(t) = psi_k(0) * exp(-i E_k t / hbar)
with E_k = (hbar^2 k^2)/(2m). We compute psi(x,t) via FFT/IFFT.

The script keeps several initial-condition presets (adapted for periodic BC):
 - "2x": psi(x)=2*x (periodic continuation; may be discontinuous at boundaries)
 - "piecewise": piecewise linear bump
 - "gauss": centered Gaussian
 - "eigen1", "eigen2": plane-wave modes with k = 2*pi*n/a

Produces PNG/GIF outputs of |psi(x,t)|^2 and prints probability diagnostics.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib import animation


def initial_psi(x, a, choice):
    # define raw (unnormalized) initial functions matching the user's examples
    psi = np.zeros_like(x, dtype=float)
    if choice == "2x":
        psi = 2.0 * x
    elif choice == "piecewise":
        # psi = x+5 for 0<x<a/2 ; psi = a/2 + 5 - x for a/2<x<a
        left = (x > 0) & (x < a / 2)
        right = (x >= a / 2) & (x < a)
        psi[left] = x[left] + 5.0
        psi[right] = a / 2.0 + 5.0 - x[right]
    elif choice == "xplus5_half":
        # convenience alias matching user's notation
        left = (x > 0) & (x < a / 2)
        right = (x >= a / 2) & (x < a)
        psi[left] = x[left] + 5.0
        psi[right] = a / 2.0 + 5.0 - x[right]
    else:
        # default: small Gaussian in the center
        xc = 0.5 * a
        sigma = 0.08 * a
        psi = np.exp(-0.5 * ((x - xc) / sigma) ** 2)
    # for periodic/free-particle we do not zero the endpoints; x is assumed
    # to be on [0,a) (endpoint=False) when called from main
    # return complex normalized wavefunction
    psi_c = psi.astype(complex)
    # use Riemann-sum normalization consistent with FFT: integral ≈ dx * sum(|ψ|^2)
    dx = x[1] - x[0]
    norm = dx * np.sum(np.abs(psi_c) ** 2)
    if norm == 0:
        return psi_c
    return psi_c / np.sqrt(norm)


def psi_at_times_fft(psi0, x, times, hbar=1.0, m=1.0):
    """Propagate initial psi0 on grid x for the list of times using FFT.

    Assumes x is uniformly spaced on [0,a) (endpoint=False) so that FFT
    corresponds to periodic boundary conditions.
    """
    Nx = len(x)
    dx = x[1] - x[0]
    # domain length a
    a = dx * Nx

    # angular wavenumbers consistent with numpy.fft (rad/m)
    k = 2.0 * pi * np.fft.fftfreq(Nx, d=dx)

    psi_k0 = np.fft.fft(psi0)
    psi_t = np.zeros((len(times), Nx), dtype=complex)
    prefactor = (1j * hbar)  # used to form exp(-i E t / hbar)
    # energy in k: E_k = (hbar^2 k^2)/(2m)
    Ek_over_hbar = (hbar * (k ** 2)) / (2.0 * m)
    for idx, t in enumerate(times):
        phase = np.exp(-1j * Ek_over_hbar * t)
        psi_k_t = psi_k0 * phase
        psi_x_t = np.fft.ifft(psi_k_t)
        psi_t[idx] = psi_x_t
    return psi_t


def main():
    parser = argparse.ArgumentParser(description="Particle in a box time evolution demo")
    parser.add_argument("--a", type=float, default=1.0, help="box width a")
    parser.add_argument("--Nx", type=int, default=800, help="number of spatial grid points")
    parser.add_argument("--N", type=int, default=120, help="number of energy eigenstates to use")
    parser.add_argument("--times", type=float, nargs="*", default=[0.0, 0.01, 0.05, 0.1, 0.2], help="times at which to evaluate (units with hbar=1,m=1)")
    parser.add_argument("--tstart", type=float, default=None, help="start time for smooth animation (overrides times when --frames used)")
    parser.add_argument("--tend", type=float, default=None, help="end time for smooth animation (overrides times when --frames used)")
    parser.add_argument("--frames", type=int, default=0, help="number of frames to generate for animation (if >0 will create linspace between tstart and tend)")
    parser.add_argument("--fps", type=int, default=8, help="frames per second for saved animation (PillowWriter)")
    preset_choices = ["2x", "piecewise", "gauss", "eigen1", "eigen2", "superpos12", "all"]
    parser.add_argument("--init", type=str, default="2x", choices=preset_choices, help="initial wavefunction choice")
    parser.add_argument("--out", type=str, default="sim/pw_time_evolution.png", help="output image file")
    parser.add_argument("--animate", action='store_true', help="save animation (GIF) of |psi|^2 over provided times")
    args = parser.parse_args()

    a = args.a
    Nx = args.Nx
    N = args.N
    # build time array: if frames provided, create linspace between tstart and tend
    if args.frames and args.frames > 0:
        t0 = args.tstart if args.tstart is not None else min(args.times)
        t1 = args.tend if args.tend is not None else max(args.times)
        times = np.linspace(t0, t1, args.frames)
    else:
        times = np.array(args.times)

    # For FFT-based propagation we prefer x on [0,a) so the grid wraps periodically.
    x = np.linspace(0.0, a, Nx, endpoint=False)

    # presets adapted for periodic BC. 'eigen' presets are simple plane waves
    # with k = 2*pi*n / a (periodic modes)
    def plane_wave(n):
        k = 2.0 * pi * n / a
        return np.exp(1j * k * x)

    presets = {
        '2x': lambda: initial_psi(x, a, '2x'),
        'piecewise': lambda: initial_psi(x, a, 'piecewise'),
        'gauss': lambda: initial_psi(x, a, 'gauss'),
        'eigen1': lambda: plane_wave(1).astype(complex),
        'eigen2': lambda: plane_wave(2).astype(complex),
        'superpos12': lambda: (plane_wave(1) + plane_wave(2)).astype(complex) / np.sqrt(2.0),
    }

    def run_for_preset(name, psi0_local):
        print(f"Running preset: {name}")
        psi0 = psi0_local

        # report norm of psi0 (use dx * sum for consistency with FFT)
        dx = x[1] - x[0]
        norm0 = dx * np.sum(np.abs(psi0) ** 2)
        print(f"Initial normalization (∫|ψ|^2 dx) = {norm0:.8f}")

        # propagate via FFT-based free-particle propagator
        psi_t = psi_at_times_fft(psi0, x, times)

        # probability conservation check and plotting
        probs = [dx * np.sum(np.abs(psi_t[i]) ** 2) for i in range(len(times))]
        for t, p in zip(times, probs):
            print(f"t={t:.6g}: ∫|ψ|^2 dx = {p:.8f}")

        save_base = args.out
        if args.animate:
            out_name = f"sim/pw_{name}.gif"
            fig, ax = plt.subplots(figsize=(6, 4))
            line, = ax.plot([], [], lw=2)
            ax.set_xlim(0, a)
            ax.set_ylim(0, np.max(np.abs(psi_t) ** 2) * 1.1)
            ax.set_xlabel('x')
            ax.set_title('|ψ(x,t)|^2')

            def init():
                line.set_data([], [])
                return (line,)

            def update(i):
                y = np.abs(psi_t[i]) ** 2
                line.set_data(x, y)
                ax.set_title(f"|ψ(x,t)|^2 at t={times[i]:.6g}")
                return (line,)

            anim = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True)
            try:
                writer = animation.PillowWriter(fps=args.fps)
                anim.save(out_name, writer=writer)
                print(f"Saved animation GIF to {out_name}")
            except Exception as e:
                print(f"Failed to save GIF for {name}: {e}")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, np.abs(psi_t[0]) ** 2)
            ax.set_xlim(0, a)
            ax.set_xlabel('x')
            out_name = f"sim/pw_{name}.png"
            fig.savefig(out_name, dpi=150)
            print(f"Saved static image to {out_name}")

    # handle running for one or all presets
    if args.init == 'all':
        for name, fn in presets.items():
            run_for_preset(name, fn())
        return


    psi0 = presets[args.init]() if args.init in presets else initial_psi(x, a, args.init if args.init != 'gauss' else 'gauss')

    print(f"Using box a={a}, Nx={Nx}")

    psi_t = psi_at_times_fft(psi0, x, times)

    # probability conservation check and plotting
    probs = [np.trapezoid(np.abs(psi_t[i]) ** 2, x) for i in range(len(times))]
    for t, p in zip(times, probs):
        print(f"t={t:.6g}: ∫|ψ|^2 dx = {p:.8f}")

    if not args.animate:
        # plotting static multipanel
        ncols = min(3, len(times))
        nrows = (len(times) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        axes = axes.flatten()
        for i, t in enumerate(times):
            ax = axes[i]
            ax.plot(x, np.abs(psi_t[i]) ** 2, color='C0')
            ax.set_title(f"|ψ(x,t)|^2 at t={t:.4g}")
            ax.set_xlabel('x')
            ax.set_xlim(0, a)
        # hide unused axes
        for j in range(len(times), len(axes)):
            axes[j].axis('off')

        fig.tight_layout()
        out = args.out
        fig.savefig(out, dpi=150)
        print(f"Saved figure to {out}")
    else:
        # create animation over the provided times and save as GIF
        fig, ax = plt.subplots(figsize=(6, 4))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, a)
        ax.set_ylim(0, np.max(np.abs(psi_t) ** 2) * 1.1)
        ax.set_xlabel('x')
        ax.set_title('|ψ(x,t)|^2')

        def init():
            line.set_data([], [])
            return (line,)

        def update(i):
            y = np.abs(psi_t[i]) ** 2
            line.set_data(x, y)
            ax.set_title(f"|ψ(x,t)|^2 at t={times[i]:.6g}")
            return (line,)

        anim = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True)
        out = args.out
        if out.lower().endswith('.gif'):
            try:
                writer = animation.PillowWriter(fps=args.fps)
                anim.save(out, writer=writer)
                print(f"Saved animation GIF to {out}")
            except Exception as e:
                print(f"Failed to save GIF via PillowWriter: {e}")
                # try mp4 fallback
                try:
                    writer = animation.FFMpegWriter(fps=args.fps)
                    out_mp4 = out.rsplit('.', 1)[0] + '.mp4'
                    anim.save(out_mp4, writer=writer)
                    print(f"Saved animation MP4 to {out_mp4}")
                except Exception as e2:
                    print(f"Failed to save MP4 via FFMpegWriter: {e2}")
        else:
            # if output doesn't end with gif, try to save mp4
            try:
                writer = animation.FFMpegWriter(fps=args.fps)
                anim.save(out, writer=writer)
                print(f"Saved animation MP4 to {out}")
            except Exception as e:
                print(f"Failed to save animation: {e}")


if __name__ == '__main__':
    main()

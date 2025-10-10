#!/usr/bin/env python3
"""
Time evolution of an electron in an infinite square well [0, a].

Uses expansion in energy eigenstates (sin basis) and computes
psi(x,t) = sum_n c_n phi_n(x) exp(-i E_n t / hbar).

Three example initial conditions are provided (match user's request):
 - "2x": psi(x)=2*x on [0,a]
 - "piecewise": psi(x)=x+5 for 0<x<a/2, and a/2+5-x for a/2<x<a
 - "custom": user-provided small lambda in the code

Produces a multi-panel PNG showing |psi(x,t)|^2 at selected times and
prints probability conservation diagnostics.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib import animation


def eigenfunc(n, x, a):
    return np.sqrt(2.0 / a) * np.sin(n * pi * x / a)


def energy(n, a, hbar=1.0, m=1.0):
    return (n ** 2) * (pi ** 2) * (hbar ** 2) / (2.0 * m * a ** 2)


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
    # set zero outside (0,a)
    psi[(x <= 0) | (x >= a)] = 0.0
    # return complex normalized wavefunction
    psi_c = psi.astype(complex)
    norm = np.trapezoid(np.abs(psi_c) ** 2, x)
    if norm == 0:
        return psi_c
    return psi_c / np.sqrt(norm)


def project_coefficients(psi0, x, a, N):
    # compute c_n = <phi_n | psi0>
    c = np.zeros(N, dtype=complex)
    for n in range(1, N + 1):
        phi_n = eigenfunc(n, x, a)
        c[n - 1] = np.trapezoid(np.conjugate(phi_n) * psi0, x)
    return c


def psi_at_times(c, x, a, times, hbar=1.0, m=1.0):
    N = len(c)
    psi_t = np.zeros((len(times), len(x)), dtype=complex)
    for idx, t in enumerate(times):
        s = np.zeros_like(x, dtype=complex)
        for n in range(1, N + 1):
            En = energy(n, a, hbar=hbar, m=m)
            phi_n = eigenfunc(n, x, a)
            s += c[n - 1] * phi_n * np.exp(-1j * En * t / hbar)
        psi_t[idx] = s
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

    x = np.linspace(0.0, a, Nx)

    presets = {
        '2x': lambda: initial_psi(x, a, '2x'),
        'piecewise': lambda: initial_psi(x, a, 'piecewise'),
        'gauss': lambda: initial_psi(x, a, 'gauss'),
        'eigen1': lambda: eigenfunc(1, x, a).astype(complex),
        'eigen2': lambda: eigenfunc(2, x, a).astype(complex),
        'superpos12': lambda: (eigenfunc(1, x, a) + eigenfunc(2, x, a)).astype(complex) / np.sqrt(2.0),
    }

    def run_for_preset(name, psi0_local):
        print(f"Running preset: {name}")
        psi0 = psi0_local
        # project
        c = project_coefficients(psi0, x, a, N)

        # report norm of psi0 and sum |c_n|^2
        norm0 = np.trapezoid(np.abs(psi0) ** 2, x)
        prob_modes = np.sum(np.abs(c) ** 2)
        print(f"Initial normalization (∫|ψ|^2 dx) = {norm0:.8f}")
        print(f"Sum |c_n|^2 over {N} modes = {prob_modes:.8f}")

        psi_t = psi_at_times(c, x, a, times)

        # probability conservation check and plotting
        probs = [np.trapezoid(np.abs(psi_t[i]) ** 2, x) for i in range(len(times))]
        for t, p in zip(times, probs):
            print(f"t={t:.6g}: ∫|ψ|^2 dx = {p:.8f}")

        save_base = args.out
        if args.animate:
            # if animate and out endswith .gif, replace base name
            base = save_base
            if save_base.endswith('.gif') or save_base.endswith('.mp4'):
                base = save_base.rsplit('.', 1)[0]
            out_name = f"sim/pw_{name}.gif"
            # reuse animation code block
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
            # static plot per preset
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

    print(f"Using box a={a}, Nx={Nx}, N_modes={N}")
    # project
    c = project_coefficients(psi0, x, a, N)

    # report norm of psi0 and sum |c_n|^2
    norm0 = np.trapezoid(np.abs(psi0) ** 2, x)
    prob_modes = np.sum(np.abs(c) ** 2)
    print(f"Initial normalization (∫|ψ|^2 dx) = {norm0:.8f}")
    print(f"Sum |c_n|^2 over {N} modes = {prob_modes:.8f}")

    psi_t = psi_at_times(c, x, a, times)

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

"""Pulse timing diagram of a compiled schedule (needs matplotlib)."""
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE, BAND = "#22221f", "#6e6d66", "#e4e3dd", "#fcfcfb", "#f1f0ea"


def _steps(pulses, key, total):
    """Piecewise-constant trace in MHz (amplitude / 2 pi) on [0, total]."""
    times, values = [0.], [0.]
    for p in sorted(pulses, key=lambda p: p["start_ns"]):
        v = 1e3*p.get(key, 0.)/(2*np.pi)
        times += [p["start_ns"], p["start_ns"], p["end_ns"], p["end_ns"]]
        values += [0., v, v, 0.]
    return times+[total], values+[0.]


def plot_schedule(schedule, path, *, title=None):
    """Rows: Omega_x, Omega_y of every qubit, then g of every coupled pair.

    Markers on a qubit row: triangles are ideal Z rotations, diamonds are
    rotating-frame changes of a tunable qubit at the edges of an exchange gate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n, total = schedule["n_qubits"], schedule["total_duration_ns"]
    pulses = schedule["pulses"]
    pairs = sorted({tuple(p["qubits"]) for p in pulses if "g" in p})
    rows = n+len(pairs)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(12, 1.05*rows+1.5),
                             facecolor=SURFACE)
    drive = max((abs(1e3*p.get(k, 0.)/(2*np.pi)) for p in pulses
                 for k in ("omega_x", "omega_y")), default=1.) or 1.
    coupling = max((1e3*p["g"]/(2*np.pi) for p in pulses if "g" in p), default=1.)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        for j, layer in enumerate(schedule["layers"]):
            if j % 2 == 0:
                ax.axvspan(layer["start_ns"], layer["end_ns"], color=BAND, lw=0, zorder=0)
            ax.axvline(layer["end_ns"], color=MUTED, lw=.8, ls=(0, (4, 3)), zorder=1)
    for q in range(n):
        ax = axes[q]
        mine = [p for p in pulses if p["qubits"] == [q]]
        for key, color in (("omega_x", BLUE), ("omega_y", ORANGE)):
            t, v = _steps(mine, key, total)
            ax.fill_between(t, v, color=color, alpha=.18, lw=0, zorder=2)
            ax.plot(t, v, color=color, lw=2, zorder=3)
        ax.set_ylim(-1.25*drive, 1.25*drive)
        ax.set_ylabel(f"Q{q+1}", rotation=0, ha="right", va="center", color=INK, fontsize=11)
        z = [e["time_ns"] for e in schedule["ideal_z"] if e["qubit"] == q]
        ax.plot(z, [-1.02*drive]*len(z), "^", color=INK, ms=6, clip_on=False, zorder=4)
        f = [e["time_ns"] for e in schedule["frame_changes"] if e["qubit"] == q]
        ax.plot(f, [1.02*drive]*len(f), "D", color=MUTED, ms=4.5, clip_on=False, zorder=4)
    for row, pair in enumerate(pairs, start=n):
        ax = axes[row]
        t, v = _steps([p for p in pulses if tuple(p["qubits"]) == pair], "g", total)
        ax.fill_between(t, v, color=AQUA, alpha=.18, lw=0, zorder=2)
        ax.plot(t, v, color=AQUA, lw=2, zorder=3)
        ax.set_ylim(-.15*coupling, 1.35*coupling)
        ax.set_ylabel(f"g {pair[0]+1}{pair[1]+1}", rotation=0, ha="right", va="center",
                      color=INK, fontsize=11)
    for j, layer in enumerate(schedule["layers"]):
        axes[0].text((layer["start_ns"]+layer["end_ns"])/2, 1.55*drive,
                     f"Layer {j+1}", ha="center", va="bottom", color=INK, fontsize=10)
    axes[-1].set_xlim(0, total)
    axes[-1].set_xlabel("time (ns)", color=INK)
    handles = [Line2D([], [], color=BLUE, lw=2, label="Ωx / 2π (MHz)"),
               Line2D([], [], color=ORANGE, lw=2, label="Ωy / 2π (MHz)"),
               Line2D([], [], color=AQUA, lw=2, label="g / 2π (MHz)"),
               Line2D([], [], color=INK, marker="^", ls="", ms=6, label="ideal Z"),
               Line2D([], [], color=MUTED, marker="D", ls="", ms=4.5, label="frame change (compensated)"
                      if (schedule.get("frames") or {}).get("compensate", True) else "frame change")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, .955), labelcolor=INK)
    fig.suptitle(title or "Pulse schedule", color=INK, fontsize=13, y=.995)
    fig.subplots_adjust(left=.07, right=.985, top=.86, bottom=.08, hspace=.35)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path

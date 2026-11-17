"""Mitigation error against circuit length, from the output of scan_steps.sh.

    python scripts/plot_steps.py results/steps_sig0.001

Reads <root>/steps<N>/mitigation.json, prints a table, writes <root>/steps.csv
and, if matplotlib is installed, <root>/steps.png (use --pdf for a PDF too).

The error of a method is |<O>_method - <O>_ideal|. By default the largest one
over the observables of the analysis is plotted; --metric rms takes their root
mean square, --observable ZIII a single one.
"""
import argparse
import csv
import json
import re
from pathlib import Path

GRAY, ORANGE, BLUE = "#7a7974", "#eb6834", "#2a78d6"
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]


def error(values, ideal, words, metric):
    errors = [abs(values[w]-ideal[w]) for w in words]
    return max(errors) if metric == "max" else (sum(e*e for e in errors)/len(errors))**.5


def collect(root, metric, observable, order):
    rows = []
    for path in root.glob("steps*/mitigation.json"):
        match = re.fullmatch(r"steps(\d+)", path.parent.name)
        if not match:
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        words = [observable] if observable else r["observables"]
        e = lambda values: error(values, r["ideal"], words, metric)
        zne, pec = r["zne"]["extrapolated"], r["pec"]
        row = {"steps": int(match[1]), "layers": r["layers"], "noisy": e(r["noisy"]),
               "ZNE SPL": e(zne["independent"][order]), "ZNE SPP": e(zne["correlated"][order]),
               "PEC SPL": e(pec["independent"]["mitigated"]),
               "Gamma SPL": pec["independent"]["gamma"]}
        for entry in pec["correlated"]:
            row[f"PEC SPP bond {entry['bond']}"] = e(entry["mitigated"])
            row[f"Gamma SPP bond {entry['bond']}"] = entry["gamma"]
        row["missing probability"] = abs(1-r["total_probability"])
        row["seconds"] = r.get("seconds", float("nan"))
        rows.append(row)
    return sorted(rows, key=lambda row: row["steps"])


def draw(rows, path, ylabel, order, pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "grid.color": "#e6e5e0", "grid.linewidth": .6})
    x = [row["steps"] for row in rows]
    bonds = [k for k in rows[0] if k.startswith("PEC SPP bond")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), constrained_layout=True)

    def line(ax, key, color, label):
        y = [row.get(key, float("nan")) for row in rows]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, mec="white", mew=1, label=label)

    line(axes[0], "noisy", GRAY, "no mitigation")
    line(axes[0], "ZNE SPL", ORANGE, "SPL (independent layers)")
    line(axes[0], "ZNE SPP", BLUE, "SPP")
    axes[0].set_title(f"ZNE, {order[:5]} {order[5:]} extrapolation", loc="left", fontsize=10)
    line(axes[1], "noisy", GRAY, "no mitigation")
    line(axes[1], "PEC SPL", ORANGE, "SPL inverse")
    for key, color in zip(bonds, BLUES[-len(bonds):] if len(bonds) <= len(BLUES) else BLUES*9):
        line(axes[1], key, color, "SPP inverse, bond "+key.split()[-1])
    axes[1].set_title("PEC", loc="left", fontsize=10)
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    limits = [ax.get_ylim() for ax in axes[:2]]
    for ax in axes[:2]:
        ax.set_ylim(min(a for a, _ in limits), max(b for _, b in limits))
    line(axes[2], "Gamma SPL", ORANGE, "SPL inverse")
    if bonds:
        last = bonds[-1].split()[-1]
        line(axes[2], f"Gamma SPP bond {last}", BLUE, f"SPP inverse, bond {last}")
    axes[2].set_title("PEC sampling overhead", loc="left", fontsize=10)
    axes[2].set_ylabel(r"$\Gamma=\sum|q|$   (shots $\times\,\Gamma^2$)")
    for ax in axes:
        ax.set_xlabel("Trotter steps")
        ax.set_xscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off() if ax is axes[2] else ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(True, axis="y")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2)
    fig.savefig(path, dpi=200)
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--metric", choices=("max", "rms"), default="max")
    parser.add_argument("--observable", help="one Pauli word instead of all observables")
    parser.add_argument("--zne-order", type=int, default=2)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args(argv)
    order = f"order{args.zne_order}"
    rows = collect(args.root, args.metric, args.observable, order)
    if not rows:
        raise SystemExit(f"no steps*/mitigation.json under {args.root}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(args.root/"steps.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)
    shown = [k for k in keys if not k.startswith("Gamma") and k not in ("steps", "layers", "seconds", "missing probability")]
    what = args.observable or f"{args.metric} over observables"
    print(f"|<O> - <O>_ideal|, {what}; ZNE {order}")
    print(f"{'steps':>6s}" + "".join(f"{k.replace('PEC SPP bond ', 'PEC SPP b'):>12s}" for k in shown)
          + f"{'1-sum(p)':>10s}{'analysis':>10s}")
    for row in rows:
        print(f"{row['steps']:6d}" + "".join(f"{row.get(k, float('nan')):12.2e}" for k in shown)
              + f"{row['missing probability']:10.1e}{row['seconds']/60:8.1f} m")
    print("1-sum(p): probability the compressed SPP had lost before it was normalised; results are "
          "only meaningful where it is far below the errors being compared")
    print(f"saved {args.root/'steps.csv'}")
    try:
        ylabel = r"$|\langle O\rangle-\langle O\rangle_{\rm ideal}|$, " + what
        draw(rows, args.root/"steps.png", ylabel, order, args.pdf)
        print(f"saved {args.root/'steps.png'}")
    except ImportError:
        print("matplotlib is not installed here: copy steps.csv (or the steps*/mitigation.json files) "
              "to a machine that has it and run this script there")


if __name__ == "__main__":
    main()

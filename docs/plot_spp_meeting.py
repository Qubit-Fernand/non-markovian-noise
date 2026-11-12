"""Render meeting figures from archived results; does not rerun simulations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
run = json.loads((ROOT / "results/spp_tn_conda_20261104/summary.json").read_text())
scan = json.loads((ROOT / "results/spp_tn_20261104_final/summary.json").read_text())
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16,
                     "text.usetex": True,
                     "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
o = run["observables"]
labels = ["Noisy", "Independent PEC", "Correlated PEC", "PEA extrapolation", "PEC: 20,000 shots"]
values = [o["noisy"], o["independent_pec"], o["correlated_pec"],
          o["pea_quadratic_extrapolation"], run["finite_shot_pec"]["estimate"]]
ax = axes[0]
ax.axvline(o["ideal"], color="#444444", ls="--", label="Ideal")
for i, (v, color) in enumerate(zip(values, ["#777777", "#ca6841", "#237566", "#6d66a8", "#237566"])):
    ax.errorbar(v, i, xerr=run["finite_shot_pec"]["standard_error"] if i == 4 else None,
                fmt="o", color=color, ms=8, capsize=4)
ax.set_yticks(range(5), labels)
ax.invert_yaxis()
ax.set_xlim(.775, 1.005)
ax.set_xlabel(r"$\langle X\rangle$")
ax.set_title("(a) Mitigation with a known noise model", fontsize=17, pad=15)
ax.legend(loc="lower left", frameon=False)
ax.grid(axis="x", alpha=.2)
ax = axes[1]
data = [r for r in scan["convergence"] if r["axis"] == "hierarchy_levels"]
ax.semilogy([r["value"] for r in data], [r["error_l2"] for r in data],
            "o-", color="#237566", lw=2, ms=8)
ax.set_xticks([3, 5, 7])
ax.set_xlabel("Retained Hermite levels")
ax.set_ylabel(r"$\|p_K-p_{10}\|_2$")
ax.set_title("(b) Hierarchy convergence", fontsize=17, pad=15)
ax.grid(alpha=.2)
fig.suptitle("One qubit, three control slots, one classical OU noise mode", fontsize=20)
fig.text(.5, .025, "Synthetic simulation; known model, no fitting. Error bar: one standard error; no learning uncertainty.",
         ha="center", fontsize=13)
fig.tight_layout(rect=(0, .08, 1, .91), w_pad=3)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"spp-meeting-20261105.{ext}", dpi=200)
print(OUT / "spp-meeting-20261105.png")

"""Generate every figure the report cites, from the committed JSON results.

Nothing here recomputes an experiment; it only plots what the run scripts wrote, so a
figure can never disagree with a number in the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Printed-size geometry: the report is A4 with 1.5 cm side margins, so \textwidth = 18.0 cm.
# Figures are drawn at exactly the size they print (width = fraction * textwidth), so a
# font of N pt in the figure is N pt on the page. Fractions must match report.tex.
TEXTWIDTH_IN = (21.0 - 2 * 1.5) / 2.54
FRAC = {"task1b_evasion.png": 1.0, "task1c_ablation.png": 1.0,
        "task2_before_after.png": 0.52}
FS = 8  # pt, minimum printed font size
FS_REPORT = 9.5  # pt, tick/label size of the two report figures (drawn at printed width)
FIG_H_IN = 1.85  # printed height of the two report figures, inches


def _style(on: bool = True, fs: float = FS):
    """8 pt print style for the three report figures; defaults for the unused others."""
    plt.rcdefaults()
    if on:
        plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs,
                             "xtick.labelsize": fs, "ytick.labelsize": fs,
                             "legend.fontsize": fs})


def printed_size(name: str, old_w: float, old_h: float, max_growth: float = 1.10):
    """Figure size in inches = printed size. Height keeps the old printed aspect ratio
    times at most max_growth; returns (w, h) and prints old vs new sizes."""
    w = FRAC[name] * TEXTWIDTH_IN
    h = w * old_h / old_w * max_growth
    print(f"{name}: old {old_w:.2f}x{old_h:.2f} in (printed {w:.2f}x{w*old_h/old_w:.2f} in, "
          f"scale {w/old_w:.3f}); new {w:.2f}x{h:.2f} in")
    return w, h


HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def evasion_curve(path: Path, out_name: str, title: str, path_after: Path | None = None):
    """Detection rate vs perturbation size, one line per attack method.

    This is the plot the lab asks for ("1B as detection rate against the size of the
    change, for each method"; "1C as the same curves again"). Left panel: the Lab 1
    model (Task 1B). Right panel, when `path_after` (results/task1c_defense.json) is
    given: the same three attacks on the same flows and the same eps grid against the
    adversarially trained model (Task 1C, `curve_after`). The ordering of the lines is the
    evidence that the weak attacks were not mistaken for the strong one.
    """
    _style(True, FS_REPORT)
    d = json.loads(path.read_text())
    n = d["n_flows"]
    panels = [(d["curve"], "Lab 1 model (before)")]
    if path_after is not None and path_after.exists():
        a = json.loads(path_after.read_text())
        # same flows and same eps grid, or the two panels are not comparable
        assert a["eval_set_provenance"] == d["eval_set_provenance"], "1B/1C flow sets differ"
        assert [r["eps_std"] for r in a["curve_after"]] == [r["eps_std"] for r in d["curve"]]
        panels.append((a["curve_after"], "adversarially trained, 2 rounds"))
    # Printed width = FRAC * textwidth; height is that of the former single-panel figure
    # (0.46 textwidth, 6.2x4.0 aspect, +10%) scaled by 0.9, since the legend now sits in a
    # strip above the panels instead of inside a 2x-tall y-range.
    w = FRAC[out_name] * TEXTWIDTH_IN
    h = FIG_H_IN
    fig, axes = plt.subplots(1, len(panels), figsize=(w, h), sharey=True, squeeze=False)
    # (key, label, color, linestyle, marker, filled, size, zorder). Where FD-FGSM and PGD
    # have identical values, PGD (dashed, hollow markers) is drawn on top of FD-FGSM.
    spec = (("random_search", "random search", "#1f77b4", ":", "o", True, 4, 2),
            ("fd_fgsm", "FD-FGSM", "#d95f02", "-", "s", True, 6, 3),
            ("pgd", f"PGD-{d['pgd']['steps']}", "#111111", "--", "^", False, 7, 4))
    for ax, (curve, ptitle) in zip(axes[0], panels):
        eps = [r["eps_std"] for r in curve]
        for name, lab, col, ls, mk, filled, ms, z in spec:
            det = [r["attacks"][name]["detection_rate"] for r in curve]
            ax.plot(eps, det, linestyle=ls, marker=mk, color=col, label=lab,
                    linewidth=1.3, markersize=ms, markerfacecolor=col if filled else "none",
                    markeredgewidth=1.0, zorder=z)
        ax.set_xlabel(r"perturbation size $\varepsilon$ (std. units)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(ptitle, fontsize=FS_REPORT, pad=3)
        ax.grid(alpha=0.3, linestyle=":")
    axes[0][0].set_ylabel(f"detection rate ({n} flows)")
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, handlelength=2.2,
               columnspacing=1.2, handletextpad=0.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / out_name, dpi=300)
    plt.close(fig)
    print(f"wrote figures/{out_name} ({len(panels)} panel(s), {w:.2f}x{h:.2f} in)")


def membership_bars(path: Path):
    """MI accuracy vs the 50% guess and vs the blind control, per day.

    The blind control bar is the point of the figure: if it is at or above the real
    attack, the apparent signal is distribution shift and nothing more.
    """
    _style(False)
    d = json.loads(path.read_text())
    days = d["days"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels, target, blind, lo, hi = [], [], [], [], []
    for day in days:
        arm_t = next(a for a in day["arms"] if a["arm"].endswith("target_model"))
        arm_b = next((a for a in day["arms"] if a["arm"].endswith("blind_control")), None)
        atk = arm_t["attacks"]["confidence_threshold"]
        labels.append(day["day"])
        target.append(atk["accuracy"])
        lo.append(atk["accuracy"] - atk["ci95"][0])
        hi.append(atk["ci95"][1] - atk["accuracy"])
        blind.append(arm_b["attacks"]["confidence_threshold"]["accuracy"]
                     if arm_b else np.nan)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, target, 0.34, label="membership attack on the target model",
           color="#c0392b")
    ax.bar(x + 0.18, blind, 0.34, label="blind control (model never saw this day)",
           color="#7f8c8d")
    ax.errorbar(x - 0.18, target, yerr=[lo, hi], fmt="none", ecolor="black",
                capsize=3, linewidth=1)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1,
               label="chance (50% guess)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("membership-inference accuracy")
    ax.set_ylim(0.45, 0.62)
    ax.set_title("Task 1A: the attack does not beat its own control", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "task1a_membership.png", dpi=200)
    plt.close(fig)
    print("wrote figures/task1a_membership.png")


def redteam_bars(path: Path):
    """Before/after success rate per attack, grouped - the report's headline table as a
    figure, with the OWASP category on the axis."""
    _style()
    d = json.loads(path.read_text())
    before = {a["key"]: a for a in d["before"]}
    after = {a["key"]: a for a in d["after"]}
    keys = list(before.keys())
    fig, ax = plt.subplots(figsize=printed_size("task2_before_after.png", 7.0, 4.2))
    y = np.arange(len(keys))
    b = [before[k]["success_rate"] or 0 for k in keys]
    a = [after[k]["success_rate"] or 0 for k in keys]
    ax.barh(y - 0.2, b, 0.38, label="before guardrails", color="#222222",
            edgecolor="black")
    ax.barh(y + 0.2, a, 0.38, label="after guardrails", color="white",
            edgecolor="black", hatch="///")
    ax.set_yticks(y, [k.replace("2A-", "").replace("2B-", "") for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel(f"attack success rate ({d['runs_per_attack']} runs each)")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "task2_before_after.png", dpi=300)
    plt.close(fig)
    print("wrote figures/task2_before_after.png")


def defense_curve(path_1b: Path, path_1c: Path):
    """PGD detection rate, undefended vs adversarially trained, on one axis."""
    _style(False)
    b = json.loads(path_1b.read_text())
    a = json.loads(path_1c.read_text())
    eps = [r["eps_std"] for r in b["curve"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, style in (("random_search", "o--"), ("fd_fgsm", "s-."), ("pgd", "^-")):
        ax.plot(eps, [r["attacks"][name]["detection_rate"] for r in b["curve"]],
                style, color="#c0392b", alpha=0.55,
                label=f"{name} (before)" if name == "pgd" else None)
        ax.plot(eps, [r["attacks"][name]["detection_rate"] for r in a["curve_after"]],
                style, color="#27ae60",
                label=f"{name} (after adversarial training)")
    ax.set_xlabel(r"perturbation size $\varepsilon$ (standardized units)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Task 1C: same attacks, before and after the defense", fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "task1c_defense.png", dpi=200)
    plt.close(fig)
    print("wrote figures/task1c_defense.png")


def ablation_curves(path_1b: Path, path_abl: Path, path_rounds: Path | None = None):
    """The four 1C configs on one axis, FD-FGSM - the attack that separates them.

    Plotted from results/task1c_ablation.json only; the undefended reference line comes
    from the Task 1B file so the two figures cannot disagree about the baseline.
    """
    _style(True, FS_REPORT)
    b = json.loads(path_1b.read_text())
    d = json.loads(path_abl.read_text())
    w = FRAC["task1c_ablation.png"] * TEXTWIDTH_IN
    fig, axes = plt.subplots(1, 2, figsize=(w, FIG_H_IN), sharey=True)
    base = [r["attacks"]["fd_fgsm"]["detection_rate"] for r in b["curve"]]
    eps = [r["eps_std"] for r in b["curve"]]
    # config -> (color, linestyle, marker, legend label)
    sty = {"full_2r": ("#c0392b", "-", "s", "full data, 2 rounds"),
           "full_1r": ("#27ae60", "--", "^", "full data, 1 round"),
           "sub100k_2r": ("#2980b9", "-.", "D", "100k rows, 2 rounds"),
           "full_2r_frozen": ("#8e44ad", "-", "v", "2 rounds, frozen"),
           "full_1r_frozen": ("#d35400", "--", "o", "1 round, frozen (rec.)")}
    if path_rounds is not None and path_rounds.exists():
        d["configs"] += json.loads(path_rounds.read_text())["configs"]
    cfgs = {c["config"]: c for c in d["configs"]}
    # two panels, one per cause: left = freezing the constant features (Cause 1),
    # right = rounds and data volume (Cause 2); the undefended curve is in both
    panels = (("Cause 1: freeze the constant features", ("full_2r", "full_2r_frozen", "full_1r_frozen")),
              ("Cause 2: rounds and data volume", ("full_2r", "full_1r", "sub100k_2r")))
    for ax, (title, keys) in zip(axes, panels):
        ax.plot(eps, base, ":o", color="black", markersize=3.5, linewidth=1.5,
                label="undefended (Lab 1)")
        for k in keys:
            cfg = cfgs[k]
            col, ls, mk, lab = sty[k]
            ax.plot([r["eps_std"] for r in cfg["curve_after"]],
                    [r["attacks"]["fd_fgsm"]["detection_rate"] for r in cfg["curve_after"]],
                    linestyle=ls, marker=mk, color=col, label=lab, linewidth=1.6,
                    markersize=4.5, markerfacecolor="none" if mk in "^v" else col)
        ax.set_xlabel(r"perturbation size $\varepsilon$ (std. units)")
        ax.set_ylim(-0.03, 1.62)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, fontsize=FS_REPORT)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(loc="upper center", ncol=2, framealpha=0.95, handlelength=2.0, borderpad=0.3,
                  labelspacing=0.25, handletextpad=0.4, columnspacing=0.8, fontsize=8)
    axes[0].set_ylabel("FD-FGSM detection rate")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "task1c_ablation.png", dpi=300)
    plt.close(fig)
    print("wrote figures/task1c_ablation.png")


if __name__ == "__main__":
    if (HERE / "task1b_evasion_wednesday.json").exists():
        evasion_curve(HERE / "task1b_evasion_wednesday.json",
                      "task1b_evasion.png",
                      "Task 1B: detection vs change size, Wednesday",
                      HERE / "task1c_defense.json")
    if (HERE / "task1a_membership.json").exists():
        membership_bars(HERE / "task1a_membership.json")
    if (HERE / "task2_redteam.json").exists():
        redteam_bars(HERE / "task2_redteam.json")
    if ((HERE / "task1b_evasion_wednesday.json").exists()
            and (HERE / "task1c_defense.json").exists()):
        defense_curve(HERE / "task1b_evasion_wednesday.json",
                      HERE / "task1c_defense.json")
    if ((HERE / "task1b_evasion_wednesday.json").exists()
            and (HERE / "task1c_ablation.json").exists()):
        ablation_curves(HERE / "task1b_evasion_wednesday.json",
                        HERE / "task1c_ablation.json",
                        HERE / "task1c_ablation_rounds.json")

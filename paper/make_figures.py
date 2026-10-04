"""Figures for the paper (leakage-filtered runs only).

Run from the repository root:  uv run python paper/make_figures.py
Reads data/features/*, runs/default/*; writes paper/figures/*.pdf.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magreg.select import mag_bin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette, light mode (print). Fixed slot order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W1, W2 = 3.4, 6.6  # single / double column width, inches

feat = pd.read_parquet(ROOT / "data/features/features.parquet")
sel = pd.read_csv(ROOT / "data/features/selection.csv")
pred = pd.read_parquet(ROOT / "runs/default/predictions.parquet")


def event_preds(set_name: str) -> pd.DataFrame:
    """Event median over stations within a fold, then mean over repeats."""
    p = pred[pred["set"] == set_name]
    ev = p.groupby(["repeat", "fold", "event_id"]).agg(mag=("mag", "first"), pred=("pred", "median"))
    return ev.groupby("event_id").agg(mag=("mag", "first"), pred=("pred", "mean")).reset_index()


# ---- Fig 1: magnitude distribution before / after selection ---------------
ev_all = feat.groupby("event_id").agg(mag=("mag", "first"), n=("station", "size"))
elig = ev_all[ev_all["n"] >= 3]
b_all = np.round(2.0 + mag_bin(elig["mag"], 0.2, 2.0) * 0.2, 1).value_counts().sort_index()
b_sel = sel.groupby("mag_bin").size()
x = b_all.index.to_numpy()
fig, ax = plt.subplots(figsize=(W1, 2.3))
w = 0.085
ax.bar(x - w / 2 - 0.004, b_all.values, width=w, color=BLUE, label=f"eligible ({len(elig):,})")
ax.bar(x + w / 2 + 0.004, b_sel.reindex(x).fillna(0).values, width=w, color=ORANGE,
       label=f"selected ({len(sel):,})")
ax.set_yscale("log")
ax.set_xlabel("Catalogue magnitude (0.2-unit bins)")
ax.set_ylabel("Events")
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
fig.savefig(OUT / "fig_selection.pdf")
plt.close(fig)

# ---- Fig 2: predicted vs catalogue magnitude (shape model) ----------------
ev = event_preds("shape")
fig, ax = plt.subplots(figsize=(W1, 3.0))
ax.scatter(ev["mag"], ev["pred"], s=5, color=BLUE, alpha=0.35, linewidths=0, rasterized=True)
lim = (1.8, 6.3)
ax.plot(lim, lim, color=INK2, lw=1.0, ls="--")
ax.text(5.6, 5.85, "1:1", color=INK2, fontsize=8)
bins = np.arange(2.0, 6.5, 0.25)
g = ev.groupby(pd.cut(ev["mag"], bins))
med = g["pred"].median()
cnt = g.size()
ctr = np.array([iv.mid for iv in med.index])
ok = cnt.to_numpy() >= 10
ax.plot(ctr[ok], med.to_numpy()[ok], color=INK, lw=1.5, marker="o", ms=3.5, label="binned median (n ≥ 10)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.set_xlabel("Catalogue magnitude")
ax.set_ylabel("Predicted magnitude")
ax.legend(loc="upper left")
fig.savefig(OUT / "fig_pred_vs_true.pdf", dpi=300)
plt.close(fig)

# ---- Fig 3: bias by magnitude for model and baselines --------------------
fig, ax = plt.subplots(figsize=(W1, 2.5))
edges = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 7.0]
labels = ["2.0", "2.5", "3.0", "3.5", "4.0", "≥4.5"]
xs = np.arange(len(labels))
for name, col, lab in (("shape", BLUE, "shape features + controls"),
                       ("amplitude", AQUA, "amplitude + controls (reference)"),
                       ("ctl", ORANGE, "controls only")):
    e = event_preds(name)
    bias = (e["pred"] - e["mag"]).groupby(pd.cut(e["mag"], edges, right=False)).mean()
    ax.plot(xs, bias.to_numpy(), color=col, marker="o", ms=3.5, label=lab)
n = ev.groupby(pd.cut(ev["mag"], edges, right=False)).size().to_numpy()
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(xs, [f"{l}\n(n={k})" for l, k in zip(labels, n)])
ax.set_xlabel("Catalogue magnitude bin (lower edge)")
ax.set_ylabel("Mean error, predicted − catalogue")
ax.legend(loc="lower left")
ax.grid(axis="x", visible=False)
fig.savefig(OUT / "fig_bias.pdf")
plt.close(fig)

# ---- Fig 4: permutation importance ---------------------------------------
NICE = {
    "corner_freq_snoke_P+S": "Corner freq., P + S (joint)",
    "corner_freq_all": "All corner-freq. estimates",
    "distance_all": "All distance controls",
    "rupture_duration_lpdt": "LPDT duration measures",
    "ctl_epi_km": "Epicentral distance",
    "f_pz_qc_snoke_fc_hz": "Corner freq., P",
    "f_sh_qc_boat_rms": "Source-fit misfit, S (corr.)",
    "f_sh_qc_snoke_fc_hz": "Corner freq., S",
    "f_lpdt_t_plateau_s": "LPDT plateau time",
    "f_pz_frac_1_2": "P energy fraction 1–2 Hz",
    "f_sh_boat_rms": "Source-fit misfit, S",
    "ctl_depth_km": "Depth",
    "ctl_hypo_km": "Hypocentral distance",
    "f_envh_t_peak_rel_sp": "Envelope peak time / (S−P)",
}
con = pd.read_csv(ROOT / "runs/default/importance_concepts.csv", index_col=0)
qty = pd.read_csv(ROOT / "runs/default/importance_quantities.csv", index_col=0).head(10)
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
for ax, t, title in ((axes[0], con, "(a) Concepts, shuffled jointly"),
                     (axes[1], qty, "(b) Quantities (station value + event median)")):
    t = t.iloc[::-1]
    y = np.arange(len(t))
    ax.barh(y, t["perm_delta_mae"], height=0.62, color=BLUE, xerr=t["perm_sd"],
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, [NICE.get(i, i) for i in t.index])
    ax.set_xlabel("Increase in event-level MAE")
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_importance.pdf")
plt.close(fig)

# ---- Fig 5: apparent corner frequency vs magnitude -----------------------
d = feat[feat["event_id"].isin(sel["event_id"])]
em = d.groupby("event_id").agg(mag=("mag", "first"), P=("f_pz_qc_snoke_fc_hz", "median"),
                               S=("f_sh_qc_snoke_fc_hz", "median"))
fig, ax = plt.subplots(figsize=(W1, 2.7))
bins = np.arange(2.0, 5.01, 0.25)
for col, c, lab in (("P", BLUE, "P"), ("S", ORANGE, "S")):
    ax.scatter(em["mag"], em[col], s=3, color=c, alpha=0.15, linewidths=0, rasterized=True)
    g = em.groupby(pd.cut(em["mag"], bins))[col]
    q = g.quantile([0.25, 0.5, 0.75]).unstack()
    k = g.size().to_numpy() >= 10
    ctr = np.array([iv.mid for iv in q.index])[k]
    ax.fill_between(ctr, q[0.25].to_numpy()[k], q[0.75].to_numpy()[k], color=c, alpha=0.18, linewidth=0)
    slope = np.polyfit(em["mag"], np.log10(em[col]), 1)[0]
    ax.plot(ctr, q[0.5].to_numpy()[k], color=c, marker="o", ms=3, label=f"{lab}: median, IQR (slope {slope:+.2f})")
ref_m = np.array([2.0, 5.0])
ref = 10 ** (np.log10(em["S"][(em["mag"] >= 2.4) & (em["mag"] < 2.6)].median()) - 0.5 * (ref_m - 2.5))
ax.plot(ref_m, ref, color=INK2, ls="--", lw=1.0, label="self-similar reference (slope −0.5)")
ax.set_yscale("log")
ax.set_ylim(0.5, 40)
ax.set_xlim(1.95, 5.05)
ax.set_xlabel("Catalogue magnitude")
ax.set_ylabel("Event-median apparent $f_c$ (Hz)")
ax.legend(loc="upper right", fontsize=7)
fig.savefig(OUT / "fig_fc_scaling.pdf", dpi=300)
plt.close(fig)
print("figures ->", OUT)

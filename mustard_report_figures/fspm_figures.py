#!/usr/bin/env python3
"""
Publication figure pipeline for the GroIMP mustard-greens FSPM report.

Reads the model outputs (canopy.csv, plant.csv) exported by dataOutput.rgg and
produces print-ready figures (PNG 600 dpi + vector PDF) plus draft captions
and a numeric summary table.

Usage
-----
    python3 fspm_figures.py                       # uses ./canopy.csv, ./plant.csv
    python3 fspm_figures.py --canopy a.csv --plant b.csv --outdir figures
    python3 fspm_figures.py --validation-canopy validation_canopy.csv

Only numpy / pandas / matplotlib are required.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# pyright: reportMissingModuleSource=false
# pyright: reportMissingImports=false
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# 1. CONFIGURATION  -- edit this block only
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    # --- crop / marketing target -------------------------------------------
    water_content: float = 0.92          # Brassica juncea, 92 % water 
    target_dm_g: float = 2.34            # minimum marketable dry mass (g/plant)
    judge_on_minimum_plant: bool = True  # every head must be marketable

    # --- measured photosynthesis parameters (Methods 2.4.3) ----------------
    amax: float = 23.172196              # umol CO2 m-2 s-1
    qe: float = 0.046027                 # umol CO2 / umol PAR
    r_measured: float = 2.123532         # dark respiration, measured
    r_in_model: float = 0.0              # value currently hard-coded in the .rgg

    # --- light environment (Methods 2.4.2) ---------------------------------
    photoperiod_h: float = 16.0
    spectrum: dict = field(default_factory=lambda: {"Red": 0.49, "Green": 0.38, "Blue": 0.13})
    leaf_reflectance: dict = field(default_factory=lambda: {"Red": 0.06, "Green": 0.09, "Blue": 0.07})
    leaf_transmittance: dict = field(default_factory=lambda: {"Red": 0.08, "Green": 0.13, "Blue": 0.03})
    petiole_reflectance: dict = field(default_factory=lambda: {"Red": 0.14, "Green": 0.22, "Blue": 0.10})

    # --- LED hardware and electricity price (energy / cost analysis) -------
    # Package PPE at 100 mA mm-2 and Tj = 25 C (Pattison et al. 2018, Table 1).
    # Driver and optics reduce complete-luminaire efficacy by a further 10-15 %.
    led_ppe: dict = field(default_factory=lambda: {
        "Warm white, 2700 K": 2.6,
        "Cool white, 6500 K": 2.9,
        "Red, 660 nm": 4.5})
    led_ppe_default: float = 2.9          # cool white, used for single-line plots
    electricity_eur_kwh: float = 0.1764  # Destatis, non-household average, H1 2026

    # --- organ expansion demo curve (Methods 2.4.1) ------------------------
    expansion_ymax: float = 1.0
    expansion_k: float = 0.30
    expansion_thalf: float = 15.0

    # --- style --------------------------------------------------------------
    font: str = "Arial"
    base_fontsize: float = 9.0
    dpi: int = 600
    single_col_in: tuple = (3.35, 2.6)   # ~85 mm
    double_col_in: tuple = (6.9, 3.0)    # ~175 mm
    save_pdf: bool = True

    @property
    def dm_fraction(self) -> float:
        return 1.0 - self.water_content

    @property
    def target_fw_g(self) -> float:
        return self.target_dm_g / self.dm_fraction


CFG = Config()

GREYS = ["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3", "#d9d9d9", "#f0f0f0"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]


def set_style(cfg: Config = CFG) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cfg.font, "Liberation Sans", "DejaVu Sans"],
        "font.size": cfg.base_fontsize,
        "axes.labelsize": cfg.base_fontsize,
        "axes.titlesize": cfg.base_fontsize,
        "xtick.labelsize": cfg.base_fontsize - 1,
        "ytick.labelsize": cfg.base_fontsize - 1,
        "legend.fontsize": cfg.base_fontsize - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
# 2. DATA LOADING
# --------------------------------------------------------------------------- #

FACTORS = ["distanceBetweenPlants_m", "power_of_light_umol_m2_s1", "totalNrPlants"]


def _tag_runs(df: pd.DataFrame, per_day_rows: int = 1) -> pd.DataFrame:
    """GroIMP appends every simulation to the same file: a new run starts at day 0."""
    df = df.copy()
    first_row_of_day0 = (df["day"] == 0)
    if per_day_rows > 1 and "plantNumber" in df:
        first_row_of_day0 &= (df["plantNumber"] == df["plantNumber"].min())
    df["run"] = first_row_of_day0.cumsum()
    return df


def load_outputs(canopy_path: Path, plant_path: Path | None, cfg: Config = CFG):
    can = pd.read_csv(canopy_path).dropna(axis=1, how="all")
    can = _tag_runs(can)

    can["spacing_m"] = can["distanceBetweenPlants_m"]
    can["ppfd"] = can["power_of_light_umol_m2_s1"]
    can["n_plants"] = can["totalNrPlants"].astype(int)
    can["density"] = can["plantDensityPerM2"]
    can["dm_g"] = can["meanTotalBiomassPerPlant_mg_plant"] / 1000.0
    can["fw_g"] = can["dm_g"] / cfg.dm_fraction
    # leaf area column is per plant in cm2 after the calcLeafArea()/updateMeanLeafArea() fix
    can["leaf_area_m2"] = can["meanLeafAreaPerPlant_cm2_plant"] / 1e4
    can["lai"] = can["leaf_area_m2"] * can["density"]
    can["yield_fw_kg_m2"] = can["fw_g"] * can["density"] / 1000.0
    can["par_mol_m2_d"] = can["ppfd"] * 3600 * cfg.photoperiod_h / 1e6
    can["label"] = can.apply(
        lambda r: f"{r.spacing_m:.2f} m / {r.ppfd:.0f} umol / n={r.n_plants}", axis=1)

    pl = None
    if plant_path is not None and Path(plant_path).exists():
        pl = pd.read_csv(plant_path).dropna(axis=1, how="all")
        pl = _tag_runs(pl, per_day_rows=2)
        pl["spacing_m"] = pl["distanceBetweenPlants_m"]
        pl["ppfd"] = pl["power_of_light_umol_m2_s1"]
        pl["n_plants"] = pl["totalNrPlants"].astype(int)
        pl["dm_g"] = pl["TotalBiomass_mg_plant"] / 1000.0
        pl["fw_g"] = pl["dm_g"] / cfg.dm_fraction
        pl["leaf_area_cm2"] = pl["LeafAreaPerPlant_cm2_plant"]
    return can, pl


def final_day_table(can: pd.DataFrame, pl: pd.DataFrame | None, cfg: Config = CFG) -> pd.DataFrame:
    last = can.loc[can.groupby("run")["day"].idxmax()].copy()
    last["yield_dm_kg_m2"] = last["dm_g"] * last["density"] / 1000.0
    cols = ["run", "spacing_m", "ppfd", "n_plants", "density", "day",
            "dm_g", "fw_g", "leaf_area_m2", "lai", "yield_fw_kg_m2", "yield_dm_kg_m2"]
    out = last[cols].rename(columns={"day": "final_day"})
    if pl is not None:
        last_day_pl = pl.loc[pl.groupby("run")["day"].transform("max") == pl["day"]]
        agg = last_day_pl.groupby("run").agg(
            fw_min_g=("fw_g", "min"), fw_max_g=("fw_g", "max"), fw_sd_g=("fw_g", "std"),
            dm_min_g=("dm_g", "min"), dm_max_g=("dm_g", "max"), dm_sd_g=("dm_g", "std"))
        out = out.merge(agg, left_on="run", right_index=True, how="left")
    judged = out["fw_min_g"] if (cfg.judge_on_minimum_plant and "fw_min_g" in out) else out["fw_g"]
    out["reaches_target"] = judged >= cfg.target_fw_g
    return out.sort_values(["ppfd", "spacing_m"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. HELPERS
# --------------------------------------------------------------------------- #


def save(fig, name: str, outdir: Path, cfg: Config = CFG) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    extra = list(fig.findobj(Legend)) 
    extra += list(fig.texts)       
    for ax in fig.axes:
        extra += list(ax.texts)      
    kw = dict(bbox_inches="tight", bbox_extra_artists=extra, pad_inches=0.05)
    fig.savefig(outdir / f"{name}.png", dpi=cfg.dpi, **kw)
    if cfg.save_pdf:
        fig.savefig(outdir / f"{name}.pdf", **kw)
    plt.close(fig)
    print(f"  saved {name}")


def broken_bar_axes(figsize, lower_top, upper_bottom, height_ratios=(3, 1)):
    """Two stacked axes emulating a broken y-axis (for near-identical bars)."""
    fig, (ax_hi, ax_lo) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.06})
    ax_hi.set_ylim(upper_bottom, None)
    ax_lo.set_ylim(0, lower_top)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=6, linestyle="none",
              color="k", mec="k", mew=0.8, clip_on=False)
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)
    return fig, ax_hi, ax_lo


def fit_saturating(x, y, weight: str = "relative"):
    """y = ymax * (1 - exp(-(x - x0) / ik)) fitted by coarse-to-fine grid search.

    weight="relative" minimises relative residuals so that the low-light points
    are honoured as much as the high-light plateau.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = 1.0 / np.clip(y, 1e-9, None) if weight == "relative" else np.ones_like(y)
    best = None
    ymax_g = np.linspace(y.max() * 0.9, y.max() * 3.0, 60)
    ik_g = np.linspace(20, 1500, 80)
    x0_g = np.linspace(0, max(x.min(), 1) * 1.5, 25)
    for _ in range(3):
        for ym in ymax_g:
            for ik in ik_g:
                for x0 in x0_g:
                    pred = ym * (1 - np.exp(-np.clip(x - x0, 0, None) / ik))
                    sse = float(np.sum(((pred - y) * w) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, ym, ik, x0)
        _, ym, ik, x0 = best
        ymax_g = np.linspace(ym * 0.8, ym * 1.25, 25)
        ik_g = np.linspace(max(ik * 0.6, 5), ik * 1.6, 25)
        x0_g = np.linspace(max(x0 - 30, 0), x0 + 30, 15)
    sse, ym, ik, x0 = best
    pred = ym * (1 - np.exp(-np.clip(x - x0, 0, None) / ik))
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(ymax=ym, ik=ik, x0=x0, r2=1 - ss_res / ss_tot if ss_tot else np.nan,
                f=lambda xx: ym * (1 - np.exp(-np.clip(np.asarray(xx, float) - x0, 0, None) / ik)))


def crossing(fx, target, lo=1.0, hi=3000.0):
    """Smallest x with fx(x) >= target (bisection); None if never reached."""
    if fx(hi) < target:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fx(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def light_response(i, amax, qe, r):
    return amax * (1 - np.exp(-qe * np.asarray(i, float) / amax)) - r


def lighting_energy_kwh_m2(ppfd, days, ppe, cfg: Config = CFG):
    """Electricity used by the lamps above 1 m2 during one production cycle."""
    watt_m2 = np.asarray(ppfd, float) / ppe
    return watt_m2 * cfg.photoperiod_h * np.asarray(days, float) / 1000.0


def lighting_cost_per_kg(ppfd, days, ppe, mass_g_plant, spacing_m, cfg: Config = CFG):
    """Lighting electricity cost per kg of fresh or dry produce."""
    cost_m2 = lighting_energy_kwh_m2(ppfd, days, ppe, cfg) * cfg.electricity_eur_kwh
    yield_kg_m2 = np.asarray(mass_g_plant, float) / 1000.0 / np.asarray(spacing_m, float) ** 2
    return cost_m2 / np.clip(yield_kg_m2, 1e-9, None)


# --------------------------------------------------------------------------- #
# 4. RESULTS FIGURES
# --------------------------------------------------------------------------- #


def fig_growth_trajectories(can, outdir, cfg=CFG, name="Fig_R1_growth_trajectories",
                            dry_mass=False):
    """Mean fresh or dry mass per plant over time; marker = spacing, color = PPFD."""
    mass_col = "dm_g" if dry_mass else "fw_g"
    target = cfg.target_dm_g if dry_mass else cfg.target_fw_g
    fig, ax = plt.subplots(figsize=cfg.double_col_in)
    spacings = sorted(can["spacing_m"].unique())
    ppfds = sorted(can["ppfd"].unique())
    mk = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(spacings)}
    palette = [color for i, color in enumerate(plt.get_cmap("tab10").colors) if i != 7]
    col = {p: palette[i % len(palette)] for i, p in enumerate(ppfds)}
    ls = {p: LINESTYLES[i % len(LINESTYLES)] if len(ppfds) <= 4 else "-"
          for i, p in enumerate(ppfds)}
    for _, g in can.groupby("run"):
        g = g.sort_values("day")
        s, p = g["spacing_m"].iloc[0], g["ppfd"].iloc[0]
        every = max(len(g) // 12, 1)
        ax.plot(g["day"], g[mass_col], color=col[p], linestyle=ls[p], marker=mk[s],
                markevery=every, markerfacecolor="w", markeredgewidth=0.8,
                markeredgecolor=col[p])
    ax.axhline(target, color="k", lw=0.9, ls=(0, (6, 3)))
    mass_label = "dry mass" if dry_mass else "fresh weight"
    target_label = f"marketable target = {target:.2f} g DM" if dry_mass else \
        f"marketable target = {target:.1f} g FW"
    ax.annotate(target_label,
                xy=(can["day"].max() * 0.02, target), xytext=(0, 3),
                textcoords="offset points", fontsize=cfg.base_fontsize - 1, va="bottom")
    ax.set_xlabel("Days after sowing")
    ax.set_ylabel(f"Mean {mass_label} per plant (g)")
    ax.set_xlim(-1, can["day"].max()+1.5)#R1
    ax.set_ylim(bottom=0)
    h1 = [Line2D([], [], color="k", marker=mk[s], ls="none", mfc="w", label=f"{s:.2f} × {s:.2f}")
          for s in spacings]
    h2 = [Line2D([], [], color=col[p], ls=ls[p], label=f"{p:.0f}") for p in ppfds]
    leg1 = ax.legend(handles=h1, title="Spacing (m)", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=h2, title="PPFD (µmol m$^{-2}$ s$^{-1}$)", loc="upper left",
              bbox_to_anchor=(1.01, 0.45), alignment="left")
    save(fig, name, outdir, cfg)


def fig_scenario_spacing(tab, pl, outdir, cfg=CFG, name="Fig_R2_spacing_selection",
                         dry_mass=False):
    """(A) per-plant mass vs spacing, (B) yield per m2."""
    mass_col = "dm_g" if dry_mass else "fw_g"
    sd_col = "dm_sd_g" if dry_mass else "fw_sd_g"
    min_col = "dm_min_g" if dry_mass else "fw_min_g"
    yield_col = "yield_dm_kg_m2" if dry_mass else "yield_fw_kg_m2"
    target = cfg.target_dm_g if dry_mass else cfg.target_fw_g
    ppfd = tab.groupby("ppfd")["spacing_m"].nunique().idxmax()
    sub = tab[(tab["ppfd"] == ppfd) & (tab["n_plants"] == tab["n_plants"].max())]
    sub = sub.drop_duplicates("spacing_m", keep="last").sort_values("spacing_m")
    if len(sub) < 2:
        print("  skipped spacing figure (needs >=2 spacings at one PPFD)")
        return
    x = np.arange(len(sub))
    labels = [f"{s:.2f}×{s:.2f}" for s in sub["spacing_m"]]

    fig = plt.figure(figsize=(cfg.double_col_in[0], 3.2))
    gs = fig.add_gridspec(1, 2, wspace=0.32)
    ax_hi, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    err = sub[sd_col].fillna(0.0).to_numpy() if sd_col in sub else np.zeros(len(sub))
    ax_hi.bar(x, sub[mass_col], width=0.62, color="0.72", edgecolor="k", linewidth=0.7,
              yerr=err, capsize=2.5, error_kw=dict(lw=0.7))
    if min_col in sub:
        ax_hi.plot(x, sub[min_col], ls="none", marker="_", ms=12, mew=1.2, color="k")
    ax_hi.axhline(target, color="k", lw=1.0, ls=(0, (5, 2.5)))
    target_label = f"{target:.2f} g" if dry_mass else f"{target:.1f} g"
    ax_hi.annotate(target_label, xy=(0, target), xytext=(2, 2),
                   textcoords="offset points", va="bottom", ha="left",
                   fontsize=cfg.base_fontsize - 1)
    ymax = max(float((sub[mass_col].to_numpy() + err).max()), target) * 1.12
    ax_hi.set_ylim(0, ymax)
    ax_hi.set_xticks(x); ax_hi.set_xticklabels(labels)
    ax_hi.set_xlabel("Spacing (m)")
    ax_hi.set_ylabel("Dry mass per plant (g)" if dry_mass else "Fresh weight per plant (g)")
    ax_hi.set_title("(A)", loc="left", fontweight="bold")

    axb.bar(x, sub[yield_col], width=0.62, color="0.45", edgecolor="k", linewidth=0.7)
    axb.set_xticks(x); axb.set_xticklabels(labels, rotation=0)
    axb.set_xlabel("Spacing (m)")
    axb.set_ylabel("Dry yield (kg m$^{-2}$)" if dry_mass else "Fresh yield (kg m$^{-2}$)")
    axb.set_title("(B)", loc="left", fontweight="bold")
    for xi, v, d in zip(x, sub[yield_col], sub["density"]):
        axb.annotate(f"{d:.0f} pl m$^{{-2}}$", xy=(xi, v), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=cfg.base_fontsize - 2)
    fig.suptitle(f"PPFD = {ppfd:.0f} µmol m$^{{-2}}$ s$^{{-1}}$", x=0.01, ha="left",
                 fontsize=cfg.base_fontsize, y=1.02)
    save(fig, name, outdir, cfg)


def fig_scenario_light(tab, outdir, cfg=CFG, name="Fig_R3_light_response", dry_mass=False):
    """(A) fresh/dry mass vs PPFD, (B) light-use efficiency."""
    mass_col = "dm_g" if dry_mass else "fw_g"
    sd_col = "dm_sd_g" if dry_mass else "fw_sd_g"
    target = cfg.target_dm_g if dry_mass else cfg.target_fw_g
    sp = tab.groupby("spacing_m")["ppfd"].nunique().idxmax()
    sub = tab[(tab["spacing_m"] == sp) & (tab["n_plants"] == tab["n_plants"].max())]
    sub = sub.sort_values("ppfd")
    if len(sub) < 3:
        print("  skipped light figure (needs >=3 PPFD levels)")
        return None
    fit = fit_saturating(sub["ppfd"], sub[mass_col])
    ppfd_req = crossing(fit["f"], target)
    bracket = None
    below = sub[sub[mass_col] < target]
    above = sub[sub[mass_col] >= target]
    if len(below) and len(above):
        b, a = below.iloc[-1], above.iloc[0]
        interp = b["ppfd"] + (a["ppfd"] - b["ppfd"]) * \
            (target - b[mass_col]) / (a[mass_col] - b[mass_col])
        bracket = (b["ppfd"], a["ppfd"], interp)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(cfg.double_col_in[0], 2.9))
    xx = np.linspace(0, sub["ppfd"].max() * 1.05, 400)
    ax.plot(xx, fit["f"](xx), color="k", lw=1.0)
    yerr = sub["fw_sd_g"].fillna(0).values if "fw_sd_g" in sub else None
    yerr = sub[sd_col].fillna(0).values if sd_col in sub else None
    ax.errorbar(sub["ppfd"], sub[mass_col], yerr=yerr, ls="none", marker="o",
                mfc="w", mec="k", color="k", capsize=2.5, elinewidth=0.7)
    ax.axhline(target, color="k", lw=0.9, ls=(0, (5, 2.5)))
    if ppfd_req:
        ax.plot([ppfd_req, ppfd_req], [0, target], color="k", lw=0.8, ls=":")
        ax.annotate(f"{ppfd_req:.0f} µmol m$^{{-2}}$ s$^{{-1}}$\nrequired for {target:.2f} g" if dry_mass
                    else f"{ppfd_req:.0f} µmol m$^{{-2}}$ s$^{{-1}}$\nrequired for {target:.1f} g",
                    xy=(ppfd_req, target), xytext=(14, -26), textcoords="offset points",
                    fontsize=cfg.base_fontsize - 1,
                    arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("PPFD (µmol m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Dry mass per plant (g)" if dry_mass else "Fresh weight per plant (g)")
    ax.set_xlim(-20, sub["ppfd"].max() * 1.10); ax.set_ylim(bottom=0)
    ax.set_title("(A)", loc="left", fontweight="bold", pad=24)

    days_ref = float(sub["final_day"].iloc[-1])
    k = (cfg.photoperiod_h * days_ref / 1000.0 / cfg.led_ppe_default
         * cfg.electricity_eur_kwh)
    secax = ax.secondary_xaxis("top", functions=(lambda p: p * k, lambda c: c / k))
    secax.set_xlabel("Lighting electricity cost (€ m$^{-2}$ cycle$^{-1}$)",
                     fontsize=cfg.base_fontsize - 1, labelpad=2)
    secax.tick_params(labelsize=cfg.base_fontsize - 1)
    ax.text(0.97, 0.06, f"$R^2$ = {fit['r2']:.3f}", transform=ax.transAxes, ha="right")

    # marginal + cumulative light-use efficiency
    par = sub["ppfd"].values * 3600 * cfg.photoperiod_h / 1e6 * sub["final_day"].values
    lue = sub["dm_g"].values * 1000.0 / par  # mg DM per mol PAR incident on the plant area
    ax2.plot(sub["ppfd"], lue, color="k", marker="s", mfc="w")
    ax2.set_xlabel("PPFD (µmol m$^{-2}$ s$^{-1}$)")
    ax2.set_ylabel("Light-use efficiency\n(mg DM per mol PAR)")
    ax2.set_title("(B)", loc="left", fontweight="bold", pad=24)
    ax2.set_ylim(bottom=0)
    save(fig, name, outdir, cfg)
    return dict(spacing=sp, fit=fit, ppfd_required=ppfd_req, bracket=bracket)


def fig_partitioning(can, outdir, cfg=CFG, name="Fig_R4_biomass_partitioning"):
    """Stacked organ biomass over time for the reference treatment."""
    ref = can[can["ppfd"] == can["ppfd"].median()]
    if ref.empty:
        ref = can
    run = ref["run"].iloc[0]
    g = can[can["run"] == run].sort_values("day")
    parts = [("meanBiomassLeafBlade_mg_plant", "Leaf blade", "0.85"),
             ("meanBiomassPetiole_mg_plant", "Petiole", "0.62"),
             ("meanBiomassStem_mg_plant", "Stem", "0.40"),
             ("meanBiomassRoot_mg_plant", "Root", "0.18")]
    parts = [p for p in parts if p[0] in g]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(cfg.double_col_in[0], 2.7))
    ys = [g[c].values / 1000.0 for c, _, _ in parts]
    ax.stackplot(g["day"], *ys, colors=[c for *_, c in parts],
                 labels=[l for _, l, _ in parts], edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Days after sowing"); ax.set_ylabel("Dry mass per plant (g)")
    ax.set_xlim(-1, g["day"].max() + 1.5); ax.set_ylim(bottom=0)#R4
    ax.legend(loc="upper left", ncol=1)
    ax.set_title("(A)", loc="left", fontweight="bold")
    tot = np.sum(ys, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = [np.where(tot > 0, y / tot, np.nan) for y in ys]
    ax2.stackplot(g["day"], *frac, colors=[c for *_, c in parts], edgecolor="k", linewidth=0.4)
    ax2.set_xlabel("Days after sowing"); ax2.set_ylabel("Fraction of total dry mass")
    ax2.set_xlim(g["day"].min() + 2, g["day"].max()); ax2.set_ylim(0, 1)
    ax2.set_title("(B)", loc="left", fontweight="bold")
    fig.suptitle(f"{g['spacing_m'].iloc[0]:.2f} m, {g['ppfd'].iloc[0]:.0f} µmol m$^{{-2}}$ s$^{{-1}}$",
                 x=0.01, ha="left", fontsize=cfg.base_fontsize, y=1.03)
    save(fig, name, outdir, cfg)


def fig_canopy_development(can, outdir, cfg=CFG, name="Fig_R5_leaf_area_LAI"):
    """Leaf area per plant and LAI over time for every run."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(cfg.double_col_in[0], 3.2))
    runs = sorted(can["run"].unique())
    for i, run in enumerate(runs):
        g = can[can["run"] == run].sort_values("day")
        lab = f"{g['spacing_m'].iloc[0]:.2f} m / {g['ppfd'].iloc[0]:.0f}"
        style = dict(color="k", ls=LINESTYLES[i % len(LINESTYLES)],
                     marker=MARKERS[i % len(MARKERS)], markevery=max(len(g) // 8, 1),
                     mfc="w", mew=0.7)
        ax.plot(g["day"], g["leaf_area_m2"] * 1e4, **style, label=lab)
        ax2.plot(g["day"], g["lai"], **style)
    ax.set_xlabel("Days after sowing"); ax.set_ylabel("Leaf area per plant (cm$^2$)")
    ax2.set_xlabel("Days after sowing"); ax2.set_ylabel("Leaf area index (m$^2$ m$^{-2}$)")
    for a, t in ((ax, "(A)"), (ax2, "(B)")):
        a.set_xlim(-1, can["day"].max() + 1.5); a.set_ylim(bottom=0)
        a.set_title(t, loc="left", fontweight="bold")
    if len(runs) <= 8:
        ax.legend(title="Spacing / PPFD", loc="upper left", fontsize=cfg.base_fontsize - 2)
    else:
        fig.subplots_adjust(bottom=0.38)
        fig.legend(handles=ax.lines, labels=[line.get_label() for line in ax.lines],
                   title="Spacing / PPFD", loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=4, fontsize=cfg.base_fontsize - 2)
    save(fig, name, outdir, cfg)


def fig_plant_variability(pl, outdir, cfg=CFG, name="Fig_R6_plant_variability", dry_mass=False):
    """Individual plants at harvest: does every head clear the marketable threshold?"""
    if pl is None:
        return
    mass_col = "dm_g" if dry_mass else "fw_g"
    target = cfg.target_dm_g if dry_mass else cfg.target_fw_g
    last = pl[pl["day"] == pl["day"].max()]
    keys = last.groupby("run").agg(sp=("spacing_m", "first"), pf=("ppfd", "first")).reset_index()
    if len(keys) < 2:
        return
    fig, ax = plt.subplots(figsize=(cfg.double_col_in[0], 2.7))
    xs, labels = [], []
    for i, (_, k) in enumerate(keys.iterrows()):
        d = last[last["run"] == k["run"]]
        jitter = np.linspace(-0.16, 0.16, len(d))
        ax.plot(np.full(len(d), i) + jitter, d[mass_col], ls="none", marker="o",
                mfc="w", mec="k", mew=0.8, ms=4)
        ax.plot([i - 0.28, i + 0.28], [d[mass_col].mean()] * 2, color="k", lw=1.4)
        xs.append(i); labels.append(f"{k['sp']:.2f} m\n{k['pf']:.0f}")
    ax.axhline(target, color="k", lw=0.9, ls=(0, (5, 2.5)))
    target_label = f" {target:.2f} g" if dry_mass else f" {target:.1f} g"
    ax.text(len(keys) - 0.5, target, target_label, va="center",
            fontsize=cfg.base_fontsize - 1)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("Spacing / PPFD (µmol m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Dry mass per plant (g)" if dry_mass else "Fresh weight per plant (g)")
    ax.set_ylim(bottom=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="w", mec="k",
                              label="individual plant"),
                       Line2D([], [], color="k", lw=1.4, label="treatment mean")],
              loc="lower right")
    save(fig, name, outdir, cfg)


def fig_competition_validation(val_can, outdir, cfg=CFG, name="Fig_R7_competition_validation",
                               dry_mass=False):
    """Isolated plant vs 2x2 stand at identical PPFD: magnitude of light competition."""
    if val_can is None or val_can["n_plants"].nunique() < 2:
        return
    mass_col = "dm_g" if dry_mass else "fw_g"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(cfg.double_col_in[0], 2.7))
    ref = {}
    for n, mk, ls in ((1, "o", "-"), (val_can["n_plants"].max(), "s", "--")):
        g = val_can[val_can["n_plants"] == n]
        if g.empty:
            continue
        g = g.groupby("day", as_index=False)[mass_col].mean()
        ref[n] = g.set_index("day")[mass_col]
        ax.plot(g["day"], g[mass_col], color="k", ls=ls, marker=mk, markevery=4, mfc="w",
                label=f"{n} plant" + ("s" if n > 1 else ""))
    ax.set_xlabel("Days after sowing")
    ax.set_ylabel("Dry mass per plant (g)" if dry_mass else "Fresh weight per plant (g)")
    ax.set_xlim(-1, val_can["day"].max() + 1.5); ax.set_ylim(bottom=0)#R7
    ax.legend(loc="upper left"); ax.set_title("(A)", loc="left", fontweight="bold")
    ns = sorted(ref)
    if len(ns) >= 2:
        rel = 100 * (1 - ref[ns[-1]] / ref[ns[0]])
        ax2.plot(rel.index, rel.values, color="k")
        ax2.axhline(0, color="0.6", lw=0.7)
        ax2.set_xlabel("Days after sowing")
        ax2.set_ylabel("Growth reduction by\nneighbour competition (%)")
        ax2.set_xlim(-1, val_can["day"].max() + 1.5)#R7
        ax2.set_title("(B)", loc="left", fontweight="bold")
        ax2.annotate(f"{rel.dropna().iloc[-1]:.1f} % at harvest",
                     xy=(rel.index[-1], rel.dropna().iloc[-1]), xytext=(-6, -14),
                     textcoords="offset points", ha="right", fontsize=cfg.base_fontsize - 1)
    save(fig, name, outdir, cfg)


def fig_energy_cost(tab, outdir, cfg=CFG, name="Fig_R8_lighting_cost", dry_mass=False):
    """Lighting electricity cost per kg fresh or dry mass for each LED type."""
    sp = tab.groupby("spacing_m")["ppfd"].nunique().idxmax()
    sub = tab[(tab["spacing_m"] == sp) & (tab["n_plants"] == tab["n_plants"].max())]
    sub = sub.sort_values("ppfd")
    if len(sub) < 3:
        print("  skipped cost figure (needs >=3 PPFD levels)")
        return None
    days = float(sub["final_day"].iloc[-1])
    x = sub["ppfd"].values
    mass_col = "dm_g" if dry_mass else "fw_g"
    mass_name = "dry mass" if dry_mass else "fresh weight"
    led_series = [(lab, ppe) for lab, ppe in cfg.led_ppe.items()
                  if lab != "Red, 660 nm"]
    greys = [str(round(g, 3)) for g in np.linspace(0.0, 0.5, len(led_series))]

    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    for (lab, ppe), gc in zip(led_series, greys):
        cost = lighting_cost_per_kg(x, days, ppe, sub[mass_col].values, sp, cfg)
        ax.plot(x, cost, color=gc, lw=1.0, marker="s", ms=3.5, mfc="w", mew=0.8,
                label=f"{lab}, {ppe:.1f} µmol J$^{{-1}}$")
    c_def = lighting_cost_per_kg(x, days, cfg.led_ppe_default, sub[mass_col].values, sp, cfg)
    j = int(np.argmin(c_def))
    ax.set_xlabel("PPFD (µmol m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel(f"Lighting electricity cost\n(€ per kg {mass_name})")
    ax.set_xlim(-20, x.max() * 1.10); ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=cfg.base_fontsize - 2, frameon=False,
              handlelength=1.8, borderaxespad=0.3)
    save(fig, name, outdir, cfg)
    return dict(spacing=sp, ppfd_opt=float(x[j]), cost_min=float(c_def[j]),
                kwh_m2=float(lighting_energy_kwh_m2(x[j], days, cfg.led_ppe_default, cfg)))


# --------------------------------------------------------------------------- #
# 5. INTRODUCTION / METHODS FIGURES
# --------------------------------------------------------------------------- #


def fig_photosynthesis_curve(outdir, cfg=CFG, name="Fig_M1_light_response_leaf"):
    """Measured leaf light-response curve, compensation point, and the r = 0 model version."""
    i = np.linspace(0, 1200, 600)
    fig, ax = plt.subplots(figsize=cfg.single_col_in)
    ax.plot(i, light_response(i, cfg.amax, cfg.qe, cfg.r_measured), color="k",
            label=f"measured ($r$ = {cfg.r_measured:.2f})")
    ax.plot(i, light_response(i, cfg.amax, cfg.qe, cfg.r_in_model), color="k", ls="--",
            label=f"as implemented ($r$ = {cfg.r_in_model:.2f})")
    ax.axhline(cfg.amax, color="0.6", lw=0.7, ls=":")
    ax.text(1190, cfg.amax, f"$A_{{max}}$ = {cfg.amax:.1f}", ha="right", va="bottom",
            fontsize=cfg.base_fontsize - 1)
    ax.axhline(0, color="0.6", lw=0.7)
    lcp = -(cfg.amax / cfg.qe) * np.log(1 - cfg.r_measured / cfg.amax)
    ax.plot([lcp], [0], marker="o", color="k", ms=4)
    ax.annotate(f"light compensation point,\n{lcp:.0f} µmol m$^{{-2}}$ s$^{{-1}}$",
                xy=(lcp, 0), xytext=(12, 104), textcoords="offset points",
                fontsize=cfg.base_fontsize - 1, arrowprops=dict(arrowstyle="->", lw=0.6))
    # initial slope = quantum efficiency
    ax.plot(i[:120], cfg.qe * i[:120] - cfg.r_measured, color="0.45", lw=0.8, ls="-.")
    ax.set_xlabel("Absorbed PAR (µmol m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Net photosynthesis\n(µmol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.set_xlim(0, 1200); ax.set_ylim(-4, cfg.amax * 1.12)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.45", lw=0.8, ls="-.",
                          label=f"initial slope $\\varphi$ = {cfg.qe:.3f}"))
    ax.legend(handles=handles, loc="lower right", fontsize=cfg.base_fontsize - 2)
    save(fig, name, outdir, cfg)
    return lcp


def fig_optical_properties(outdir, cfg=CFG, name="Fig_M2_optical_properties"):
    """(A) spectral composition of the lamp, (B) leaf reflectance/transmittance/absorptance."""
    bands = list(cfg.spectrum)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(cfg.double_col_in[0], 2.6))
    band_grey = {"Red": "0.30", "Green": "0.60", "Blue": "0.82"}
    ax.bar(bands, [cfg.spectrum[b] for b in bands],
           color=[band_grey[b] for b in bands], edgecolor="k", lw=0.7, width=0.6)
    for b in bands:
        ax.annotate(f"{cfg.spectrum[b]:.2f}", xy=(b, cfg.spectrum[b]), xytext=(0, 2),
                    textcoords="offset points", ha="center")
    ax.set_ylabel("Fraction of emitted PAR"); ax.set_ylim(0, 0.6)
    ax.set_title("(A)", loc="left", fontweight="bold")

    refl = np.array([cfg.leaf_reflectance[b] for b in bands])
    tran = np.array([cfg.leaf_transmittance[b] for b in bands])
    absb = 1 - refl - tran
    x = np.arange(len(bands))
    ax2.bar(x, absb, color="0.25", edgecolor="k", lw=0.7, width=0.6, label="absorbed")
    ax2.bar(x, refl, bottom=absb, color="0.65", edgecolor="k", lw=0.7, width=0.6, label="reflected")
    ax2.bar(x, tran, bottom=absb + refl, color="0.92", edgecolor="k", lw=0.7, width=0.6,
            label="transmitted")
    for xi, a in zip(x, absb):
        ax2.annotate(f"{a:.2f}", xy=(xi, a / 2), ha="center", va="center", color="w",
                     fontsize=cfg.base_fontsize - 1)
    ax2.set_xticks(x); ax2.set_xticklabels(bands)
    ax2.set_ylabel("Fraction of incident light"); ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right", ncol=1)
    ax2.set_title("(B)", loc="left", fontweight="bold")
    save(fig, name, outdir, cfg)


def fig_organ_expansion(outdir, cfg=CFG, name="Fig_M3_organ_expansion"):
    """Logistic organ-expansion function with ymax, k and thalf annotated."""
    t = np.linspace(0, 40, 400)
    y = cfg.expansion_ymax / (1 + np.exp(-cfg.expansion_k * (t - cfg.expansion_thalf)))
    fig, ax = plt.subplots(figsize=cfg.single_col_in)
    ax.plot(t, y, color="k")
    ax.axhline(cfg.expansion_ymax, color="0.6", lw=0.7, ls=":")
    ax.text(39.5, cfg.expansion_ymax, "$y_{max}$", ha="right", va="bottom")
    ax.axhline(cfg.expansion_ymax / 2, color="0.6", lw=0.7, ls=":")
    ax.axvline(cfg.expansion_thalf, color="0.6", lw=0.7, ls=":")
    ax.plot([cfg.expansion_thalf], [cfg.expansion_ymax / 2], marker="o", color="k", ms=4)
    ax.annotate("$t_{half}$", xy=(cfg.expansion_thalf, 0), xytext=(3, 6),
                textcoords="offset points")
    sl = cfg.expansion_k * cfg.expansion_ymax / 4
    tt = np.linspace(cfg.expansion_thalf - 6, cfg.expansion_thalf + 6, 20)
    ax.plot(tt, cfg.expansion_ymax / 2 + sl * (tt - cfg.expansion_thalf), color="0.45",
            lw=0.8, ls="-.")
    ax.annotate(f"slope at $t_{{half}}$ = $k\\,y_{{max}}$/4  ($k$ = {cfg.expansion_k})",
                xy=(cfg.expansion_thalf + 6, cfg.expansion_ymax / 2 + sl * 6),
                xytext=(4, -2), textcoords="offset points", fontsize=cfg.base_fontsize - 2)
    ax.set_xlabel("Organ age (days)")
    ax.set_ylabel("Relative organ size $y$($t$)/$y_{max}$")
    ax.set_xlim(0, 40); ax.set_ylim(0, cfg.expansion_ymax * 1.15)
    save(fig, name, outdir, cfg)


def fig_design_to_scale(tab, outdir, cfg=CFG, name="Fig_M4_experimental_design"):
    """Top view of the 2x2 stand drawn to scale for every spacing, with LAI at harvest."""
    ppfd = tab.groupby("ppfd")["spacing_m"].nunique().idxmax()
    sub = (tab[tab["ppfd"] == ppfd]
           .drop_duplicates("spacing_m", keep="last").sort_values("spacing_m"))
    if sub.empty:
        return
    n = len(sub)
    fig, axes = plt.subplots(1, n, figsize=(cfg.double_col_in[0], 2.2))
    axes = np.atleast_1d(axes)
    smax = sub["spacing_m"].max()
    for ax, (_, row) in zip(axes, sub.iterrows()):
        s = row["spacing_m"]
        lim = 2 * smax * 1.22
        ax.set_xlim(-lim / 2, lim / 2); ax.set_ylim(-lim / 2, lim / 2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        for dx in (-0.5, 0.5):
            for dy in (-0.5, 0.5):
                ax.add_patch(Rectangle((dx * s - s / 2, dy * s - s / 2), s, s,
                                       facecolor="0.90", edgecolor="0.45", lw=0.6))
                ax.plot([dx * s], [dy * s], marker="+", color="k", ms=5, mew=1.0)
        y0 = -lim / 2 * 0.88
        ax.plot([-s, 0], [y0] * 2, color="k", lw=1.4)
        ax.text(-s / 2, y0 + lim * 0.03, f"{s:.2f} m", ha="center",
                fontsize=cfg.base_fontsize - 2)
        ax.set_title(f"{s:.2f} × {s:.2f} m\n{row['density']:.0f} plants m$^{{-2}}$\n"
                     f"LAI = {row['lai']:.1f}", fontsize=cfg.base_fontsize - 1)
    fig.suptitle(f"2 × 2 stand, PPFD = {ppfd:.0f} µmol m$^{{-2}}$ s$^{{-1}}$, LAI at final day",
                 x=0.01, ha="left", fontsize=cfg.base_fontsize, y=1.06)
    save(fig, name, outdir, cfg)


# --------------------------------------------------------------------------- #
# 6. TEXT OUTPUTS
# --------------------------------------------------------------------------- #

CAPTIONS = """# Draft figure captions

**Figure R1: Fresh weight increase per plant.**
Mean fresh weight (g) per simulated *Brassica juncea* var. *rugosa* plant from day 0 to {maxday},
for {n_sp} plant spacings and {n_pf} light intensities (2 x 2 stand). Marker shapes identify
spacing and line colors identify PPFD. Fresh weight was derived from simulated dry mass assuming
a water content of {wc:.0f} %. The dashed line marks the minimum marketable fresh weight of
{target:.1f} g per plant.

**Figure R2: Spacing selection (Scenario I).**
(A) Fresh weight per plant at day {maxday} for each spacing at PPFD = {ppfd_sc1:.0f} umol m-2 s-1.
Bars are treatment means of four plants, error bars are standard deviations between plants and
horizontal ticks show the lightest individual plant. The dashed line is the marketable threshold
({target:.1f} g FW).
(B) Fresh yield per unit area for the same treatments.

**Figure R3: Light response and light-use efficiency (Scenario II).**
(A) Fresh weight per plant at day {maxday} as a function of PPFD at {sp_sc2:.2f} x {sp_sc2:.2f} m
spacing. The line is a saturating fit, y = ymax (1 - exp(-(I - I0)/Ik)); the dotted line marks the
PPFD required to reach {target:.1f} g FW per plant. (B) Light-use efficiency, expressed as dry
mass produced per mol of PAR delivered to the ground area of one plant over the whole cycle.

**Figure R4: Biomass partitioning.**
(A) Absolute and (B) relative dry-mass partitioning between leaf blade, petiole, stem and root
over the simulated cycle for the reference treatment.

**Figure R5: Canopy development.**
(A) Leaf area per plant and (B) leaf area index over time for all simulated treatments.

**Figure R6: Variability between individual plants at harvest.**
Fresh weight of each of the four simulated plants per treatment at day {maxday}; horizontal bars
are treatment means, the dashed line is the marketable threshold. Positional differences arise
from stochastic phyllotaxis and Monte-Carlo ray tracing in the light model.

**Figure R7: Verification of light competition.**
(A) Fresh weight per plant for an isolated plant and for a 2 x 2 stand simulated at identical PPFD
and spacing. (B) Relative growth reduction caused by neighbour competition over time. This
comparison verifies that the light model resolves plant-plant shading.

**Figure R8: Lighting electricity cost per kilogram of fresh weight.**
Lighting cost as a function of PPFD for three LED photon efficacies. The vertical line and
annotation indicate the cost minimum for the default LED efficacy.

Dry-mass counterparts of the mass-based results figures are saved separately with the `_DM`
suffix. They use dry mass per plant, dry-mass yield per area, and the 2.34 g DM target.

**Figure M1: Leaf light-response curve used for parameterisation.**
Net photosynthesis as a function of absorbed PAR, following
A = Amax (1 - exp(-phi I / Amax)) - r, with Amax = {amax:.2f} umol CO2 m-2 s-1,
phi = {qe:.4f} umol CO2 / umol PAR and r = {rm:.2f} umol CO2 m-2 s-1 measured by gas exchange.
The dashed curve shows the behaviour of the currently implemented code (r = {r0:.2f}), which
removes the light compensation point at {lcp:.0f} umol m-2 s-1.

**Figure M2: Optical properties used in the light model.**
(A) Spectral composition of the simulated lamp. (B) Modelled fraction of incident light that is
absorbed, reflected and transmitted by a leaf per waveband. Green light is absorbed least, which
is consistent with the absorbance spectrum of chlorophyll.

**Figure M3: Organ expansion function.**
Logistic expansion y(t) = ymax / (1 + exp(-k (t - thalf))) used for all organs, with the
parameters ymax, k and thalf indicated.

**Figure M4: Simulated stand layouts, drawn to scale.**
Ground area per plant for each spacing treatment (2 x 2 arrangement, crosses = plant positions),
with plant density and the leaf area index reached at the final day.
"""


def write_scenario_cost_table(tab, outdir, cfg=CFG):
    reference_spacing = tab.groupby("spacing_m")["ppfd"].nunique().idxmax()
    reference_ppfd = tab.groupby("ppfd")["spacing_m"].nunique().idxmax()
    selected = tab[(tab["spacing_m"] == reference_spacing) | (tab["ppfd"] == reference_ppfd)]
    selected = (selected.drop_duplicates(["spacing_m", "ppfd"], keep="last")
                .sort_values(["spacing_m", "ppfd"]))
    table = pd.DataFrame({
        "spacing_m": selected["spacing_m"].to_numpy(),
        "ppfd": selected["ppfd"].to_numpy(),
        "fw_g": selected["fw_g"].to_numpy(),
        "smallest_plant_g": selected["fw_min_g"].to_numpy(),
        "yield_fw_kg_m2": selected["yield_fw_kg_m2"].to_numpy(),
        "cost_eur_kg_fw": [
            lighting_cost_per_kg(row.ppfd, row.final_day, cfg.led_ppe_default,
                                 row.fw_g, row.spacing_m, cfg)
            for row in selected.itertuples()],
        "reaches_target": selected["reaches_target"].to_numpy(),
    })
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "scenario_cost_table.csv", index=False)

    lines = [
        "| Spacing (m) | PPFD | FW per plant (g) | Smallest plant (g) | Yield (kg FW m^-2) | Cost (EUR kg^-1 FW) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        spacing = f"{row.spacing_m:.2f} x {row.spacing_m:.2f}"
        smallest = f"{row.smallest_plant_g:.1f}" + (" *" if row.reaches_target else "")
        lines.append(f"| {spacing} | {row.ppfd:.0f} | {row.fw_g:.1f} | {smallest} "
                     f"| {row.yield_fw_kg_m2:.2f} | {row.cost_eur_kg_fw:.2f} |")
    lines.extend([
        "",
        f"Cost assumes {cfg.led_ppe_default:.1f} umol/J LED efficacy, "
        f"{cfg.electricity_eur_kwh:.4f} EUR/kWh, and each treatment's simulated final day.",
        f"* Minimum individual FW meets the {cfg.target_fw_g:.2f} g marketable target.",
    ])
    (outdir / "scenario_cost_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    headers = ["Spacing (m)", "PPFD", "FW / plant (g)", "Smallest (g)",
               "Yield (kg FW m$^{-2}$)", "Cost (EUR kg$^{-1}$ FW)"]
    cells = [[f"{row.spacing_m:.2f} x {row.spacing_m:.2f}", f"{row.ppfd:.0f}",
              f"{row.fw_g:.1f}",
              f"{row.smallest_plant_g:.1f}" + (" *" if row.reaches_target else ""),
              f"{row.yield_fw_kg_m2:.2f}", f"{row.cost_eur_kg_fw:.2f}"]
             for row in table.itertuples()]
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axis("off")
    ax.set_title("Treatment summary at harvest", loc="left", pad=12,
                 fontsize=cfg.base_fontsize + 2, fontweight="bold")
    table_artist = ax.table(cellText=cells, colLabels=headers, cellLoc="center",
                            colLoc="center", bbox=[0, 0.18, 1, 0.76],
                            colWidths=[0.18, 0.10, 0.15, 0.16, 0.21, 0.20])
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(cfg.base_fontsize)
    for (row_idx, _), cell in table_artist.get_celld().items():
        cell.set_edgecolor("0.78")
        cell.set_linewidth(0.55)
        if row_idx == 0:
            cell.set_facecolor("0.20")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row_idx % 2 else "0.94")
    ax.text(0, 0.10,
            f"Cost: PPE {cfg.led_ppe_default:.1f} umol/J; "
            f"electricity {cfg.electricity_eur_kwh:.4f} EUR/kWh.",
            transform=ax.transAxes, ha="left", va="center", fontsize=cfg.base_fontsize - 1)
    ax.text(0, 0.04,
            f"* Smallest individual FW meets the {cfg.target_fw_g:.2f} g marketable target.",
            transform=ax.transAxes, ha="left", va="center", fontsize=cfg.base_fontsize - 1)
    fig.savefig(outdir / "scenario_cost_table.png", dpi=cfg.dpi,
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_text_outputs(tab, outdir, cfg, extras):
    write_scenario_cost_table(tab, outdir, cfg)
    tab.to_csv(outdir / "summary_final_day.csv", index=False)
    maxday = int(tab["final_day"].max())
    sc1 = tab.groupby("ppfd")["spacing_m"].nunique().idxmax()
    txt = CAPTIONS.format(
        maxday=maxday, n_sp=tab["spacing_m"].nunique(), n_pf=tab["ppfd"].nunique(),
        wc=cfg.water_content * 100, target=cfg.target_fw_g, ppfd_sc1=sc1,
        sp_sc2=extras.get("sc2_spacing", tab["spacing_m"].mode().iloc[0]),
        amax=cfg.amax, qe=cfg.qe, rm=cfg.r_measured, r0=cfg.r_in_model,
        lcp=extras.get("lcp", float("nan")))
    (outdir / "figure_captions.md").write_text(txt, encoding="utf-8")

    lines = ["Key numbers for the Results text", "=" * 34, ""]
    lines.append(f"marketable target : {cfg.target_dm_g:.2f} g DM = {cfg.target_fw_g:.2f} g FW "
                 f"(water content {cfg.water_content:.0%})")
    if "lcp" in extras:
        lines.append(f"light compensation point (measured r): {extras['lcp']:.1f} umol m-2 s-1")
    if extras.get("ppfd_required"):
        lines.append(f"PPFD required at {extras['sc2_spacing']:.2f} m spacing: "
                     f"{extras['ppfd_required']:.0f} umol m-2 s-1 (from fitted curve)")
    if extras.get("ppfd_bracket"):
        lo, hi, interp = extras["ppfd_bracket"]
        lines.append(f"  simulated bracket: {lo:.0f} umol (fail) -> {hi:.0f} umol (pass); "
                     f"linear interpolation = {interp:.0f} umol m-2 s-1")
    
    show = ["spacing_m", "ppfd", "n_plants", "dm_g", "fw_g", "fw_min_g", "lai",
            "yield_fw_kg_m2", "reaches_target"]
    show = [c for c in show if c in tab]
    lines.append(tab[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    (outdir / "key_numbers.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  saved summary_final_day.csv, scenario_cost_table.csv/.md/.png, "
          "figure_captions.md, key_numbers.txt")


# --------------------------------------------------------------------------- #
# 7. MAIN
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canopy", default="canopy.csv")
    ap.add_argument("--plant", default="plant.csv")
    ap.add_argument("--validation-canopy", default=None,
                    help="canopy.csv of the isolated-plant vs stand verification run")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    set_style(CFG)
    outdir = Path(args.outdir)
    can, pl = load_outputs(Path(args.canopy), Path(args.plant), CFG)
    tab = final_day_table(can, pl, CFG)
    val_can = None
    if args.validation_canopy:
        val_can, _ = load_outputs(Path(args.validation_canopy), None, CFG)

    print(f"Loaded {can['run'].nunique()} simulation run(s) -> {outdir}/")
    extras = {}
    fig_growth_trajectories(can, outdir, CFG)
    fig_growth_trajectories(can, outdir, CFG,
                            name="Fig_R1_growth_trajectories_DM", dry_mass=True)
    fig_scenario_spacing(tab, pl, outdir, CFG)
    fig_scenario_spacing(tab, pl, outdir, CFG,
                         name="Fig_R2_spacing_selection_DM", dry_mass=True)
    sc2 = fig_scenario_light(tab, outdir, CFG)
    fig_scenario_light(tab, outdir, CFG,
                       name="Fig_R3_light_response_DM", dry_mass=True)
    if sc2:
        extras["sc2_spacing"] = sc2["spacing"]
        extras["ppfd_required"] = sc2["ppfd_required"]
        extras["ppfd_bracket"] = sc2["bracket"]
    fig_partitioning(can, outdir, CFG)
    fig_canopy_development(can, outdir, CFG)
    fig_plant_variability(pl, outdir, CFG)
    fig_plant_variability(pl, outdir, CFG,
                          name="Fig_R6_plant_variability_DM", dry_mass=True)
    fig_competition_validation(val_can, outdir, CFG)
    fig_competition_validation(val_can, outdir, CFG,
                               name="Fig_R7_competition_validation_DM", dry_mass=True)
    ec = fig_energy_cost(tab, outdir, CFG)
    fig_energy_cost(tab, outdir, CFG, name="Fig_R8_lighting_cost_DM", dry_mass=True)
    if ec:
        extras["ppfd_cost_optimum"] = ec["ppfd_opt"]
        extras["cost_min_eur_per_kg"] = ec["cost_min"]
        extras["kwh_m2_at_optimum"] = ec["kwh_m2"]
    extras["lcp"] = fig_photosynthesis_curve(outdir, CFG)
    fig_optical_properties(outdir, CFG)
    fig_organ_expansion(outdir, CFG)
    fig_design_to_scale(tab, outdir, CFG)
    write_text_outputs(tab, outdir, CFG, extras)
    print("done.")


if __name__ == "__main__":
    main()

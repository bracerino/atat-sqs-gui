"""
Partial Radial Distribution Function (PRDF) calculation helpers.

The calculation and plotting logic here is modelled on the standalone
XRDlicious (P)RDF calculator: PRDF values are obtained from matminer's
``PartialRadialDistributionFunction`` featurizer, an optional minimum-distance
cutoff is applied, a total RDF is built by summing the per-pair contributions,
and traces can be smoothed (Gaussian / Savitzky-Golay / cubic spline),
normalised, and rendered as smooth curves, raw points, or bar histograms.

Streamlit is intentionally *not* imported here so this module stays a pure,
reusable calculation/plotting layer. The Streamlit UI lives in the caller.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from matminer.featurizers.structure import PartialRadialDistributionFunction

# --- Online-version limits -------------------------------------------------
# Keep PRDF tractable on the hosted (online) app. Compile the app locally and
# raise these values for unrestricted calculations.
MAX_PRDF_ATOMS = 500
MAX_PRDF_SPECIES = 6
# Max number of structures (converted + comparison bestsqs.out files) the online
# PRDF may process at once.
MAX_PRDF_STRUCTURES = 5


def rgb_to_hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


PRDF_COLORS = [rgb_to_hex(c) for c in plt.cm.tab10.colors]
PRDF_FONT = dict(size=18, color="black")


def get_structure_species(structure) -> set:
    """Return the set of element symbols present in a pymatgen Structure."""
    species = set()
    for site in structure:
        if site.is_ordered:
            species.add(site.specie.symbol)
        else:
            for sp in site.species:
                species.add(sp.symbol)
    return species


def validate_prdf_structure(structure, max_atoms=MAX_PRDF_ATOMS, max_species=MAX_PRDF_SPECIES):
    """Check a structure against the online PRDF limits.

    Returns ``(ok: bool, reason: str)``.
    """
    n_atoms = len(structure)
    n_sp = len(get_structure_species(structure))
    if n_atoms > max_atoms:
        return (
            False,
            f"Structure has **{n_atoms} atoms**, which exceeds the online limit of "
            f"**{max_atoms}**. Use a smaller supercell, or compile the app locally to raise the limit.",
        )
    if n_sp > max_species:
        return (
            False,
            f"Structure has **{n_sp} element types**, which exceeds the online limit of "
            f"**{max_species}**. Use a structure with fewer species, or compile the app locally.",
        )
    return (True, "")


def compute_prdf(structure, cutoff=10.0, bin_size=0.1, r_min=0.0):
    """Compute the PRDF for a single (ordered) pymatgen Structure.

    Returns a dict with:
      - ``prdf_dict``:  {(el_a, el_b): np.ndarray of g(r) values}
      - ``dist_dict``:  {(el_a, el_b): [bin-center distances]}
      - ``global_rdf``: {bin_center: summed intensity}  (total RDF)
    """
    featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
    featurizer.fit([structure])
    prdf_vals = featurizer.featurize(structure)
    labels = featurizer.feature_labels()

    prdf_dict = {}
    dist_dict = {}
    global_rdf = {}
    for j, label in enumerate(labels):
        pair_str, rng = label.split(" PRDF r=")
        pair = tuple(pair_str.split("-"))
        lo, hi = map(float, rng.split("-"))
        bc = (lo + hi) / 2.0
        if bc < r_min:
            continue
        prdf_dict.setdefault(pair, []).append(prdf_vals[j])
        dist_dict.setdefault(pair, []).append(bc)
        global_rdf[bc] = global_rdf.get(bc, 0.0) + prdf_vals[j]

    prdf_dict = {p: np.asarray(v, dtype=float) for p, v in prdf_dict.items()}
    return {"prdf_dict": prdf_dict, "dist_dict": dist_dict, "global_rdf": global_rdf}


# --- Smoothing -------------------------------------------------------------
def _norm(y):
    y = np.asarray(y, dtype=float)
    m = float(np.max(y)) if len(y) else 0.0
    return y / m if m > 0 else y


def smooth_gaussian(y, sigma=1.5):
    return gaussian_filter1d(y, sigma=sigma)


def smooth_savgol(y, window=11, polyorder=3):
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    window = max(window, 3)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(y, window, polyorder)


def smooth_spline(x, y, n_pts=300):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        return x, y
    xs = np.linspace(x[0], x[-1], n_pts)
    return (xs, np.maximum(0, make_interp_spline(x, y, k=3)(xs)))


def apply_smoothing(x_arr, y_arr, plot_style="Smooth Curve",
                    smoothing_method="Gaussian", sigma=1.5,
                    sg_win=11, sg_ord=3, spline_pts=300):
    """Return (x, y) after applying the chosen smoothing (no-op unless Smooth Curve)."""
    x_arr, y_arr = (np.asarray(x_arr, dtype=float), np.asarray(y_arr, dtype=float))
    if plot_style != "Smooth Curve":
        return (x_arr, y_arr)
    if smoothing_method == "Gaussian":
        return (x_arr, smooth_gaussian(y_arr, sigma))
    elif smoothing_method == "Savitzky-Golay":
        return (x_arr, smooth_savgol(y_arr, sg_win, sg_ord))
    else:
        return smooth_spline(x_arr, y_arr, spline_pts)


# --- Plotting --------------------------------------------------------------
def add_prdf_trace(fig, x, y, name, color, *, plot_style="Smooth Curve",
                   line_style="Lines Only", normalize=False, bin_size=0.1,
                   bar_width_factor=0.8, smoothing_method="Gaussian",
                   sigma=1.5, sg_win=11, sg_ord=3, spline_pts=300, dash="solid"):
    """Add a single (P)RDF trace to a plotly figure, honouring the plot style."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if normalize:
        y = _norm(y)

    mode = "lines" if line_style == "Lines Only" else "lines+markers"

    if plot_style == "Bars (Histogram)":
        fig.add_trace(go.Bar(
            x=x, y=y, name=name,
            marker=dict(color=color, line=dict(width=0)),
            width=bin_size * bar_width_factor, opacity=0.75,
        ))
        return

    xp, yp = apply_smoothing(x, y, plot_style, smoothing_method, sigma, sg_win, sg_ord, spline_pts)

    if plot_style == "Smooth Curve":
        # faint stems showing the raw binned values underneath the smooth curve
        x_stems, y_stems = ([], [])
        for xi, yi in zip(x, y):
            x_stems.extend([xi, xi, None])
            y_stems.extend([0, yi, None])
        fig.add_trace(go.Scatter(
            x=x_stems, y=y_stems, mode="lines", name=f"{name} (raw)",
            line=dict(color=color, width=1), opacity=0.35, showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=xp, y=yp, mode=mode, name=name,
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=7) if "markers" in mode else dict(),
    ))


def make_prdf_layout(title, normalize=False, barmode=None):
    ylabel = "Pair correlation function g(r)" if not normalize else "Normalized intensity"
    d = dict(
        title=dict(text=title, font=PRDF_FONT),
        xaxis=dict(title=dict(text="Distance (Å)", font=PRDF_FONT), tickfont=PRDF_FONT),
        yaxis=dict(title=dict(text=ylabel, font=PRDF_FONT), tickfont=PRDF_FONT, range=[0, None]),
        hovermode="x",
        font=PRDF_FONT,
        hoverlabel=dict(font=PRDF_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
                    font=dict(size=15)),
    )
    if barmode:
        d["barmode"] = barmode
    return d

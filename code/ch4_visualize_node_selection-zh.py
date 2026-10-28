"""Visualize peak / off-peak physical-node selection for the subgraph pooling.

Reads the JSON produced by ``subgraph_node_selection_stats.py`` (the file is
``subgraph_selection_<DATASET>.json``) and writes three publication-style figures:

  1. ``subgraph_selection_<DATASET>_map.png``
     Two maps: peak frequency and off-peak frequency.  Points are placed by
     sensor longitude / latitude; the color encodes the chosen frequency metric.

  2. ``subgraph_selection_<DATASET>_diff_map.png``
     The peak-minus-off-peak frequency difference on its own map.

  3. ``subgraph_selection_<DATASET>_peak_vs_offpeak.png``
     A peak-vs-off-peak scatter plot with a 1:1 reference line.  Points are
     colored by peak-off-peak bias so the spatial difference map and the
     scatter plot use the same diverging color semantics.

If ``contextily`` is installed and network tiles are available, a light
``Esri.WorldGrayCanvas`` basemap is added by default (CartoDB tiles now require
an API key, so they are not used).  Otherwise the script falls back to a clean
lat/lon grid (no basemap required).

Usage:
    python experiments/visualize_node_selection.py \
        --json results/nodesSelect-PEMS-BAY/subgraph_selection_PEMS-BAY.json

    # CartoDB Positron with an API key (apply at https://carto.com/basemaps/apikey/):
    python experiments/visualize_node_selection.py \
        --json results/nodesSelect-PEMS-BAY/subgraph_selection_PEMS-BAY.json \
        --basemap-source CartoDB.Positron \
        --basemap-api-key YOUR_CARTO_API_KEY
"""

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

PEAK_COLOR = "#c0392b"
OFFPEAK_COLOR = "#2c7fb8"

METRIC_LABELS = {
    "sample_selection_freq": "样本被选择频率（样本比例）",
    "selection_freq": "逐层选择概率",
    "selection_count": "被选择次数",
}

_TIMES_NEW_ROMAN_FONT = (
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
)
_FANGSONG_FONT = (
    "/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/fandol/"
    "FandolFang-Regular.otf"
)
font_manager.fontManager.addfont(_TIMES_NEW_ROMAN_FONT)
font_manager.fontManager.addfont(_FANGSONG_FONT)

plt.rcParams.update({
    "font.family": ["Times New Roman", "FandolFang"],
    "font.size": 9,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _arrays(data, metric):
    mapping = data["node_mapping"]
    selection = data["selection"]
    lons = np.asarray([n["longitude"] for n in mapping], dtype=float)
    lats = np.asarray([n["latitude"] for n in mapping], dtype=float)
    ids = [str(n["physical_id"]) for n in mapping]
    peak = np.asarray(selection.get("peak", {}).get(metric, []), dtype=float)
    offpeak = np.asarray(selection.get("off-peak", {}).get(metric, []), dtype=float)
    if peak.size != len(mapping):
        peak = np.zeros(len(mapping))
    if offpeak.size != len(mapping):
        offpeak = np.zeros(len(mapping))
    return ids, lons, lats, peak, offpeak


def _add_basemap(ax, source_name, timeout, api_key=None):
    import contextily as cx

    provider = cx.providers.query_name(source_name)
    token_keys = [
        key for key, val in provider.items()
        if isinstance(val, str) and "<insert your" in val and key in provider.get("url", "")
    ]
    if token_keys:
        if not api_key:
            raise RuntimeError(
                f"basemap provider '{source_name}' requires an API key; "
                "pass --basemap-api-key or set CARTO_API_KEY."
            )
        provider = provider(**{token_keys[0]: api_key})

    cx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=provider,
        attribution=False,
        timeout=timeout,
    )
    ax.set_axis_off()


def _set_geo_extent(ax, lons, lats):
    pad_x = 0.08 * (lons.max() - lons.min() + 1e-9)
    pad_y = 0.08 * (lats.max() - lats.min() + 1e-9)
    ax.set_xlim(lons.min() - pad_x, lons.max() + pad_x)
    ax.set_ylim(lats.min() - pad_y, lats.max() + pad_y)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lats.mean())))


def _format_geo_ticks(ax, lons, lats, grid=False):
    _set_geo_extent(ax, lons, lats)
    ax.set_axis_on()
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.2f}"))
    ax.tick_params(axis="both", labelsize=8)
    if grid:
        ax.grid(color="0.75", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def _format_geo_axes(ax, lons, lats):
    _format_geo_ticks(ax, lons, lats, grid=True)


def _annotate_top(ax, rank_values, k, ids, lons, lats):
    if k <= 0:
        return
    order = np.argsort(np.asarray(rank_values, dtype=float))[::-1][:k]
    for i in order:
        ax.annotate(
            ids[i],
            xy=(lons[i], lats[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.55",
                      lw=0.4, alpha=0.82),
        )


def plot_map(data, metric, out_path, dpi=300, annotate_k=6, no_basemap=False,
             basemap_source="Esri.WorldGrayCanvas", basemap_timeout=15,
             basemap_api_key=None):
    ids, lons, lats, peak, offpeak = _arrays(data, metric)
    label = METRIC_LABELS.get(metric, metric)
    dataset = data.get("dataset", "dataset")
    peak_n = data["selection"].get("peak", {}).get("n_samples", 0)
    offpeak_n = data["selection"].get("off-peak", {}).get("n_samples", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.98, bottom=0.12, wspace=0.33)

    common_min = float(np.minimum(peak.min(), offpeak.min()))
    common_max = float(np.maximum(peak.max(), offpeak.max()))
    norm = Normalize(vmin=common_min, vmax=common_max)

    for ax, vals, title in (
        (axes[0], peak, f"高峰期 (n={peak_n})"),
        (axes[1], offpeak, f"非高峰期 (n={offpeak_n})"),
    ):
        ax.scatter(
            lons, lats, c=vals, cmap="viridis", norm=norm,
            s=22, edgecolors="white", linewidths=0.35, alpha=0.9, zorder=3,
        )
        _annotate_top(ax, vals, annotate_k, ids, lons, lats)
        ax.set_title(title, fontsize=16, fontweight="bold")

    basemap_ok = False
    basemap_error = ""
    if not no_basemap:
        try:
            _add_basemap(axes[0], basemap_source, basemap_timeout, basemap_api_key)
            basemap_ok = True
            for ax in axes[1:]:
                try:
                    _add_basemap(ax, basemap_source, basemap_timeout, basemap_api_key)
                except Exception:
                    ax.set_axis_off()
        except Exception as exc:
            basemap_ok = False
            basemap_error = str(exc)

    if basemap_ok:
        for ax in axes:
            _format_geo_ticks(ax, lons, lats, grid=False)
    else:
        for ax in axes:
            _format_geo_axes(ax, lons, lats)
        print("[info] contextily basemap unavailable; using lat/lon grid instead."
              + (f" ({basemap_error})" if basemap_error else ""))

    for ax in axes:
        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(label)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    fig.suptitle(
        f"节点在高峰期与非高峰期的选择情况（{dataset}）",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_diff_map(data, metric, out_path, dpi=300, annotate_k=6, no_basemap=False,
                  basemap_source="Esri.WorldGrayCanvas", basemap_timeout=15,
                  basemap_api_key=None):
    ids, lons, lats, peak, offpeak = _arrays(data, metric)
    diff = peak - offpeak
    dataset = data.get("dataset", "dataset")

    fig, ax = plt.subplots(figsize=(6.2, 5.3), dpi=dpi)
    # fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.12)
    abs_max = float(np.max(np.abs(diff))) if diff.size else 1.0
    if abs_max == 0.0:
        abs_max = 1.0
    diff_norm = Normalize(vmin=-abs_max, vmax=abs_max)
    sc = ax.scatter(
        lons, lats, c=diff, cmap="RdBu_r", norm=diff_norm,
        s=22, edgecolors="white", linewidths=0.35, alpha=0.9, zorder=3,
    )
    _annotate_top(ax, np.abs(diff), annotate_k, ids, lons, lats)

    basemap_ok = False
    basemap_error = ""
    if not no_basemap:
        try:
            _add_basemap(ax, basemap_source, basemap_timeout, basemap_api_key)
            basemap_ok = True
        except Exception as exc:
            basemap_error = str(exc)

    if basemap_ok:
        _format_geo_ticks(ax, lons, lats, grid=False)
    else:
        _format_geo_axes(ax, lons, lats)
        print("[info] contextily basemap unavailable; using lat/lon grid instead."
              + (f" ({basemap_error})" if basemap_error else ""))

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("高峰期与非高峰期频率差")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    fig.suptitle(
        f"高峰期与非高峰期频率差（{dataset}）",
        fontsize=16,
        fontweight="bold",
        # y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_peak_offpeak_scatter(data, metric, out_path, dpi=300, annotate_k=10):
    ids, lons, lats, peak, offpeak = _arrays(data, metric)
    diff = peak - offpeak
    label = METRIC_LABELS.get(metric, metric)
    dataset = data.get("dataset", "dataset")

    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=dpi)
    lo = float(min(peak.min(), offpeak.min()))
    hi = float(max(peak.max(), offpeak.max()))
    margin = 0.03 * (hi - lo + 1e-9)
    lo -= margin
    hi += margin

    abs_max = float(np.max(np.abs(diff))) if diff.size else 1.0
    if abs_max == 0.0:
        abs_max = 1.0
    diff_norm = Normalize(vmin=-abs_max, vmax=abs_max)
    sc = ax.scatter(
        offpeak, peak, c=diff, cmap="RdBu_r", norm=diff_norm,
        s=20, edgecolors="white", linewidths=0.3, alpha=0.85,
    )
    ax.plot([lo, hi], [lo, hi], "--", color="0.45", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("非高峰期" + label.replace("\n", " "))
    ax.set_ylabel("高峰期" + label.replace("\n", " "))
    ax.grid(color="0.85", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    order = np.argsort(np.abs(diff))[::-1][:annotate_k]
    for i in order:
        ax.annotate(
            ids[i],
            xy=(offpeak[i], peak[i]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.55",
                      lw=0.4, alpha=0.82),
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("高峰期与非高峰期频率差值")
    ax.set_title("高峰期与非高峰期频率差分布（PEMS-BAY）",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize physical-node selection frequency on a map"
    )
    p.add_argument("--json", required=True,
                   help="subgraph_selection_<DATASET>.json from subgraph_node_selection_stats.py")
    p.add_argument("--metric", default="sample_selection_freq",
                   choices=list(METRIC_LABELS),
                   help="which frequency metric to encode in color")
    p.add_argument("--out", default=None,
                   help="output directory; defaults to the JSON's directory")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--annotate-top-k", type=int, default=6,
                   help="label this many top-ranked nodes per map panel")
    p.add_argument("--no-basemap", action="store_true",
                   help="skip contextily basemap even if it is available")
    p.add_argument("--basemap-source", default="Esri.WorldGrayCanvas",
                   help="contextily provider name, e.g. Esri.WorldGrayCanvas, "
                        "Esri.WorldStreetMap or OpenStreetMap.Mapnik")
    p.add_argument("--basemap-timeout", type=float, default=15,
                   help="seconds before giving up on a tile request")
    p.add_argument("--basemap-api-key", default=None,
                   help="API key/token for providers that require one (e.g. CartoDB); "
                        "also read from CARTO_API_KEY")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_data(args.json)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.json))
    os.makedirs(out_dir, exist_ok=True)
    basemap_api_key = args.basemap_api_key or os.environ.get("CARTO_API_KEY")

    dataset = data.get("dataset", "dataset")
    map_path = os.path.join(out_dir, f"subgraph_selection_{dataset}_map.png")
    diff_map_path = os.path.join(out_dir, f"subgraph_selection_{dataset}_diff_map.png")
    scatter_path = os.path.join(
        out_dir, f"subgraph_selection_{dataset}_peak_vs_offpeak.png"
    )

    plot_map(
        data,
        args.metric,
        map_path,
        dpi=args.dpi,
        annotate_k=args.annotate_top_k,
        no_basemap=args.no_basemap,
        basemap_source=args.basemap_source,
        basemap_timeout=args.basemap_timeout,
        basemap_api_key=basemap_api_key,
    )
    plot_diff_map(
        data,
        args.metric,
        diff_map_path,
        dpi=args.dpi,
        annotate_k=args.annotate_top_k,
        no_basemap=args.no_basemap,
        basemap_source=args.basemap_source,
        basemap_timeout=args.basemap_timeout,
        basemap_api_key=basemap_api_key,
    )
    plot_peak_offpeak_scatter(
        data,
        args.metric,
        scatter_path,
        dpi=args.dpi,
        annotate_k=max(args.annotate_top_k, 6),
    )


if __name__ == "__main__":
    main()

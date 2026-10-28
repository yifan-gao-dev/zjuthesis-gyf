"""Visualize the SMS-MoE ``scale_gate`` routing from ADAMS-MoE datasets.

  Reads ``routing_metrics.json`` produced by ``routing_stats.py`` from
  ``./results/routing-<DATASET>/`` and writes a high-fidelity PNG figure into the
  same directory.

  Expected ``scale_gate`` layout (as emitted by routing_stats.py):
      "scale_gate": {                            # regime -> stats, NOT nested by name
          "peak": {
              "n_samples": int,
              "mean_gate": [float, ...],         # mean weight per scale
              "mean_entropy": float,
          },
          "off-peak": {...},
          "overall": {...},
      }
  """

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from ch4_visualize_routing_zh import (
    OFFPEAK_COLOR,
    PEAK_COLOR,
    discover_json,
    find_jsons,
    load_routing,
    present_regimes,
    regime_label,
)


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
    "axes.unicode_minus": False,
})


def plot_scale_gate(dataset, scale, out_dir, dpi):
    regimes = present_regimes(scale)
    if not regimes:
        print(f"[skip] {dataset}: no peak/off-peak scale_gate data")
        return

    k = len(scale[regimes[0]]["mean_gate"])
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=dpi)

    x = np.arange(k)
    width = 0.8 / len(regimes)
    for i, regime in enumerate(regimes):
        color = PEAK_COLOR if regime == "peak" else OFFPEAK_COLOR
        offset = (i - (len(regimes) - 1) / 2.0) * width
        ax.bar(x + offset, scale[regime]["mean_gate"], width,
               label=regime_label(regime, scale), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{j}" for j in range(k)])
    ax.set_ylabel("尺度权重平均值", fontsize=14)
    ax.set_title(f"SMS-MoE 在 {dataset} 数据集上的各尺度权重", fontsize=16,
                 fontweight="bold", pad=4)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(out_dir, f"{dataset}_scale_gate.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def main():
    ap = argparse.ArgumentParser(description="Visualize SMS-MoE scale_gate routing stats")
    ap.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS-BAY", "PEMS04"])
    ap.add_argument("--results-dir", default="results",
                    help="base directory to search for routing_metrics.json")
    ap.add_argument("--dpi", type=int, default=600, help="figure DPI (300+ recommended)")
    ap.add_argument("--find", action="store_true",
                    help="list discovered routing_metrics.json files and exit")
    args = ap.parse_args()

    all_found = find_jsons(args.results_dir)
    if args.find:
        print(f"Found {len(all_found)} routing_metrics.json under '{args.results_dir}':")
        for p in all_found:
            print("  " + p)
        return

    for ds in args.datasets:
        path = discover_json(args.results_dir, ds)
        if path is None:
            print(f"[error] no routing_metrics.json matched dataset '{ds}' under "
                  f"'{args.results_dir}'")
            print("        available files:")
            for p in all_found:
                print("          " + p)
            continue
        data = load_routing(path)
        scale = data.get("scale_gate")
        if not scale:
            print(f"[skip] {ds}: no scale_gate block in '{path}'")
            continue
        out_dir = os.path.dirname(path)   # 图片和 json 放在同一目录
        print(f"[read] {path}")
        plot_scale_gate(ds, scale, out_dir, args.dpi)


if __name__ == "__main__":
    main()

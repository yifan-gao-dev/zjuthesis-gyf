"""Visualize peak vs off-peak expert routing for ADAMS-MoE datasets.

  Reads routing_metrics.json produced by routing_stats.py from
  ./results/routing-<DATASET>/ and writes high-fidelity PNG figures into the same
  directory.

  Expected JSON layout (as emitted by routing_stats.py):
      {
          "dataset": str,
          "meta": {"ckpt": str, "n_samples": int, "peak": str},
          "gates": {
              "<expert_type>": {                 # e.g. "dynamic", "static"
                  "peak": {
                      "n_samples": int,
                      "select_freq": [float, ...],          # per expert
                      "mean_gate": [float, ...],
                      "mean_gate_when_selected": [float, ...],
                      "mean_entropy": float,
                  },
                  "off-peak": {...},
                  "overall": {...},
              },
          },
      }

  The optional ``scale_gate`` block in the JSON is visualized separately by
  ``visualize_scale.py``.
  """

import argparse
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PEAK_COLOR = "#c0392b"
OFFPEAK_COLOR = "#2c7fb8"

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
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})


# def load_routing(results_dir, dataset):
#     path = os.path.join(results_dir, f"routing-{dataset}", "routing_metrics.json")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_") or "gate"


def present_regimes(stats):
    """peak/off-peak regimes that actually contain samples."""
    return [r for r in ("peak", "off-peak") if r in stats and stats[r].get("n_samples", 0) > 0]


def regime_label(regime, stats):
    ent = None
    zh_regime = None
    if regime == 'peak':
        zh_regime = '高峰期'
    else:
        zh_regime = '非高峰期'
    return zh_regime


def plot_overview(dataset, data, out_dir, dpi):
    gates = data.get("gates", {})

    parts = []  # (name, n_bins, regimes, stats)
    for gate_name, stats in gates.items():
        regimes = present_regimes(stats)
        if not regimes:
            continue
        k = len(stats[regimes[0]]["select_freq"])
        parts.append((gate_name, k, regimes, stats))

    if not parts:
        print(f"[skip] {dataset}: no peak/off-peak gate data")
        return

    fig, axes = plt.subplots(1, len(parts), figsize=(4.8 * len(parts), 4.0), dpi=dpi)
    if len(parts) == 1:
        axes = [axes]

    for ax, (name, k, regimes, stats) in zip(axes, parts):
        x = np.arange(k)
        width = 0.8 / len(regimes)
        for i, regime in enumerate(regimes):
            color = PEAK_COLOR if regime == "peak" else OFFPEAK_COLOR
            offset = (i - (len(regimes) - 1) / 2.0) * width
            ax.bar(x + offset, stats[regime]["select_freq"], width,
                   label=regime_label(regime, stats), color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f"E{j}" for j in range(k)])
        ax.set_ylabel("选择频率", fontsize=14)
        if name == 'static':
            m_name = 'SG-MoE'
        else:
            m_name = 'DyG-MoE'
        ax.set_title(f"SD-MoE/{m_name} 在 {dataset} 数据集上的专家路由", fontsize=16,
                     fontweight="bold", pad=4)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(out_dir, f"{dataset}_routing_overview.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_expert_detail(dataset, gate_name, stats, out_dir, dpi):
    regimes = present_regimes(stats)
    if not regimes:
        return
    k = len(stats[regimes[0]]["select_freq"])
    metrics = [
        ("select_freq", "选择频率"),
        ("mean_gate_when_selected", "被选择时的平均门控权重"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=dpi)
    for ax, (key, ylabel) in zip(axes, metrics):
        x = np.arange(k)
        width = 0.8 / len(regimes)
        for i, regime in enumerate(regimes):
            color = PEAK_COLOR if regime == "peak" else OFFPEAK_COLOR
            offset = (i - (len(regimes) - 1) / 2.0) * width
            ax.bar(x + offset, stats[regime][key], width,
                    label=regime_label(regime, stats), color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"E{j}" for j in range(k)])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=16, fontweight="bold", pad=4)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(out_dir, f"{dataset}_{sanitize(gate_name)}_routing.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def normalize(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_jsons(results_dir):
    found = []
    if not os.path.isdir(results_dir):
        return found
    for root, _dirs, files in os.walk(results_dir):
        if "routing_metrics.json" in files:
            found.append(os.path.join(root, "routing_metrics.json"))
    return sorted(found)


def discover_json(results_dir, dataset):
    target = normalize(dataset)
    candidates = find_jsons(results_dir)
    # 优先按 json 所在目录名匹配（忽略大小写、连字符/下划线等）
    for p in candidates:
        dirname = normalize(os.path.basename(os.path.dirname(p)))
        if target in dirname or dirname in target:
            return p
    # 再退而求其次按整条路径匹配
    for p in candidates:
        if target in normalize(p):
            return p
    return None


def load_routing(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Visualize peak/off-peak routing stats")
    ap.add_argument("--datasets", nargs="+", default=["METR-LA", "PEMS04"])
    ap.add_argument("--results-dir", default="results",
                    help="base directory to search for routing_metrics.json")
    ap.add_argument("--dpi", type=int, default=600, help="figure DPI (300+ recommended)")
    ap.add_argument("--no-detail", action="store_true",
                    help="only write the per-dataset overview figure")
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
        out_dir = os.path.dirname(path)   # 图片和 json 放在同一目录
        print(f"[read] {path}")
        plot_overview(ds, data, out_dir, args.dpi)
        if not args.no_detail:
            for gate_name, stats in data.get("gates", {}).items():
                plot_expert_detail(ds, gate_name, stats, out_dir, args.dpi)

if __name__ == "__main__":
    main()

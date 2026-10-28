import matplotlib.pyplot as plt

import numpy as np

from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter

font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/times.ttf')
font_manager.fontManager.addfont('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/fandol/FandolFang-Regular.otf')

plt.rcParams['font.family'] = ['Times New Roman', 'FandolFang']
plt.rcParams['axes.unicode_minus'] = False

# 数据
x = np.array(["1", "2", "3", "4"])

pred_96 = np.array([0.274, 0.274, 0.278, 0.275])
pred_192 = np.array([0.334, 0.329, 0.332, 0.347])
pred_336 = np.array([0.358, 0.361, 0.357, 0.360])
pred_720 = np.array([0.391, 0.390, 0.384, 0.386])

# 创建 1x4 四个子图，顺序为 96、192、336、720
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

horizons = [96, 192, 336, 720]
preds = [pred_96, pred_192, pred_336, pred_720]
colors = ['blue', 'orange', 'green', 'red']
markers = ['o', 's', '^', 'd']

ylim_by_horizon = {
    96: (0.265, 0.285),
    192: (0.320, 0.350),
    336: (0.355, 0.365),
    720: (0.380, 0.395),
}

for ax, horizon, pred, color, marker in zip(axes, horizons, preds, colors, markers):
    ax.plot(x, pred, color=color, marker=marker)
    ax.set_title(f'ETTh2-{horizon}', fontsize=18)
    ax.set_xlabel('尺度数量', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_ylim(*ylim_by_horizon[horizon])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.tight_layout()
plt.savefig('ch3-multiScaleVars.png', dpi=600, bbox_inches='tight')
plt.show()

# # 可用的标记样式
# marker_styles = plt.Line2D.markers.keys()
# marker_styles

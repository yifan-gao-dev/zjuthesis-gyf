import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/times.ttf')
font_manager.fontManager.addfont('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/fandol/FandolFang-Regular.otf')

plt.rcParams['font.family'] = ['Times New Roman', 'FandolFang']
plt.rcParams['axes.unicode_minus'] = False
fang_font = font_manager.FontProperties(
    fname='/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/fandol/FandolFang-Regular.otf'
)
# sequence: 48-96-192-336-512-720
data = {
    '本章方法': {
        'ETTh1':        {96: [0.386, 0.369, 0.363, 0.359, 0.354, 0.354], 720: [0.490, 0.475, 0.446, 0.452, 0.431, 0.423]},
        'Electricity':  {96: [0.193, 0.147, 0.131, 0.127, 0.126, 0.130], 720: [0.244, 0.216, 0.198, 0.197, 0.190, 0.189]},
        'Weather':      {96: [0.171, 0.154, 0.151, 0.150, 0.150, 0.149], 720: [0.343, 0.330, 0.312, 0.307, 0.306, 0.319]}
    },
    'iTransformer': {
        'ETTh1':        {96: [0.394, 0.387, 0.393, 0.402, 0.400, 0.396], 720: [0.522, 0.502, 0.516, 0.492, 0.492, 0.567]},
        'Electricity':  {96: [0.178, 0.148, 0.134, 0.132, 0.152, 0.133], 720: [0.241, 0.211, 0.197, 0.209, 0.194, 0.194]},
        'Weather':      {96: [0.204, 0.176, 0.169, 0.163, 0.168, 0.180], 720: [0.381, 0.358, 0.342, 0.326, 0.341, 0.352]}
    },
    'FITS': {
        'ETTh1':        {96: [0.399, 0.386, 0.384, 0.380, 0.372, 0.383], 720: [0.504, 0.466, 0.438, 0.430, 0.426, 0.433]},
        'Electricity':  {96: [0.258, 0.203, 0.164, 0.157, 0.153, 0.151], 720: [0.300, 0.258, 0.231, 0.224, 0.220, 0.218]},
        'Weather':      {96: [0.191, 0.166, 0.152, 0.146, 0.149, 0.150], 720: [0.374, 0.348, 0.330, 0.318, 0.320, 0.308]}
    },
    'SOFTS': {
        'ETTh1':        {96: [0.394, 0.382, 0.386, 0.386, 0.382, 0.389], 720: [0.545, 0.514, 0.480, 0.467, 0.499, 0.508]},
        'Electricity':  {96: [0.171, 0.145, 0.132, 0.130, 0.145, 0.133], 720: [0.227, 0.215, 0.209, 0.201, 0.215, 0.225]},
        'Weather':      {96: [0.197, 0.172, 0.170, 0.161, 0.172, 0.165], 720: [0.377, 0.361, 0.341, 0.346, 0.361, 0.387]}
    },
    'DLinear': {
        'ETTh1':        {96: [0.411, 0.397, 0.394, 0.384, 0.384, 0.384], 720: [0.548, 0.521, 0.509, 0.501, 0.504, 0.511]},
        'Electricity':  {96: [0.245, 0.195, 0.152, 0.140, 0.140, 0.133], 720: [0.278, 0.242, 0.214, 0.204, 0.204, 0.197]},
        'Weather':      {96: [0.241, 0.197, 0.185, 0.174, 0.174, 0.168], 720: [0.376, 0.348, 0.336, 0.332, 0.332, 0.315]}
    },
    'PatchTST': {
        'ETTh1':        {96: [0.410, 0.393, 0.382, 0.382, 0.370, 0.376], 720: [0.509, 0.480, 0.452, 0.449, 0.447, 0.448]},
        'Electricity':  {96: [0.218, 0.165, 0.159, 0.150, 0.143, 0.147], 720: [0.273, 0.230, 0.207, 0.201, 0.199, 0.201]},
        'Weather':      {96: [0.246, 0.179, 0.160, 0.156, 0.148, 0.157], 720: [0.386, 0.351, 0.330, 0.320, 0.313, 0.325]} 
    },
}

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
markers = ['o', 's', '+', '^', 'd', '*']

x = [48, 96, 192, 336, 512, 720]


fig, axs = plt.subplots(2, 3, figsize=(15, 9), dpi=600)  # 创建2行3列的子图

# 在每个子图上绘图
for i in range(2):
    for j in range(3):
        horizon_idx = 96 if i == 0 else 720
        if j == 0:
            dataset_idx = 'ETTh1'
        elif j == 1:
            dataset_idx = 'Electricity'
        else:
            dataset_idx = 'Weather'
        axs[i, j].plot(x, data['本章方法'][dataset_idx][horizon_idx], marker=markers[0], color=colors[0])
        axs[i, j].plot(x, data['iTransformer'][dataset_idx][horizon_idx], marker=markers[1], color=colors[1])
        axs[i, j].plot(x, data['FITS'][dataset_idx][horizon_idx], marker=markers[2], color=colors[2])
        axs[i, j].plot(x, data['SOFTS'][dataset_idx][horizon_idx], marker=markers[3], color=colors[3])
        axs[i, j].plot(x, data['DLinear'][dataset_idx][horizon_idx], marker=markers[4], color=colors[4])
        axs[i, j].plot(x, data['PatchTST'][dataset_idx][horizon_idx], marker=markers[5], color=colors[5])

# 设置标签和标题等
for ax in axs.flat:
    ax.set_xlabel('H', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    ax.grid(True)

axs[0, 0].set_title('ETTh1 F=96', fontsize=18)
axs[0, 1].set_title('Electricity F=96', fontsize=18)
axs[0, 2].set_title('Weather F=96', fontsize=18)
axs[1, 0].set_title('ETTh1 F=720', fontsize=18)
axs[1, 1].set_title('Electricity F=720', fontsize=18)
axs[1, 2].set_title('Weather F=720', fontsize=18)


handles, labels = [], []
for model, color, marker in zip(data.keys(), colors, markers):
    handles.append(plt.Line2D([0], [0], color=color, marker=marker))
    labels.append(model)

legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(data), fontsize=16)

for text, label in zip(legend.get_texts(), labels):
    if label == '本章方法':
        text.set_fontproperties(fang_font)
        text.set_fontsize(16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the legend
plt.savefig('ch3-lookBackVars.png', dpi=600, bbox_inches='tight')

plt.tight_layout()
plt.show()

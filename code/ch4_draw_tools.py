import folium
import os
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import font_manager

import seaborn as sns
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import warnings
# 屏蔽上一个警告
warnings.filterwarnings("ignore", message="Setting the 'color' property will override the edgecolor or facecolor properties.")
# 屏蔽当前的 tight_layout 警告
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

_TIMES_TTF = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
_FANGSONG_TTF = "/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/fandol/FandolFang-Regular.otf"
if os.path.exists(_TIMES_TTF):
    font_manager.fontManager.addfont(_TIMES_TTF)
if os.path.exists(_FANGSONG_TTF):
    font_manager.fontManager.addfont(_FANGSONG_TTF)


# def draw_boxplot_pic(scale_samples, ID, _color):
#     sns.set_theme(style="whitegrid", font_scale=1.6)

#     plt.figure(figsize=(12, 6))
#     ax = sns.boxplot(
#         data=scale_samples,
#         width=0.5,
#         showfliers=False,
#         medianprops=dict(color="black", linewidth=2),
#         boxprops=dict(edgecolor="black"),
#         whiskerprops=dict(color="black"),
#         capprops=dict(color="black"),
#         palette=_color
#     )

#     sns.stripplot(
#         data=scale_samples,
#         color="black",
#         size=4,
#         jitter=0.15,
#         alpha=0.6
#     )

#     # 自动 y 轴范围
#     all_vals = np.concatenate(scale_samples)
#     y_min, y_max = all_vals.min(), all_vals.max()
#     margin = 0.05 * (y_max - y_min)
#     plt.ylim(y_min - margin, y_max + margin)

#     # ---------- 字体加粗设置 ----------
#     # for metr-la
#     plt.xticks(
#         range(len(scale_samples)),
#         ["207", "51", "12"],
#         fontweight="bold"
#     )

#     plt.xlabel(
#         "Spatial Scale (#Nodes)",
#         fontweight="bold",
#         fontsize=18
#     )
#     plt.ylabel(
#         "Temporal Mean of Sensor Value",
#         fontweight="bold",
#         fontsize=18
#     )

#     # y / x 轴刻度字体加粗
#     ax.tick_params(axis="both", labelsize=16)
#     for tick in ax.get_xticklabels() + ax.get_yticklabels():
#         tick.set_fontweight("bold")
#     # ---------------------------------

#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(ID)

def draw_boxplot_pic(scale_samples, ID, _color):
    sns.set_theme(style="whitegrid", font_scale=1.6)
    plt.rcParams["font.family"] = ["Times New Roman", "FandolFang"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 6))
    # import ipdb; ipdb.set_trace()
    ax = sns.boxplot(
        data=scale_samples,
        width=0.5,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        palette=_color
    )

    sns.stripplot(
        data=scale_samples,
        color="black",
        size=4,
        jitter=0.15,
        alpha=0.6
    )

    # 自动 y 轴范围
    all_vals = np.concatenate(scale_samples)
    y_min, y_max = all_vals.min(), all_vals.max()
    margin = 0.05 * (y_max - y_min)
    plt.ylim(y_min - margin, y_max + margin)

    plt.xticks(
        range(len(scale_samples)),
        ["尺度 0", "尺度 1", "尺度 2"]
    )

    plt.title("多尺度传感器数值箱线图", fontsize=20, fontweight="bold", pad=12)
    plt.ylabel("传感器数值的时间均值", fontsize=18, fontweight="bold")
    plt.xlabel("空间尺度", fontsize=18, fontweight="bold")

    ax.tick_params(axis="both", labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    sns.despine()
    plt.tight_layout()
    plt.savefig(ID, dpi=600)

def saveImg(HTML_PATH, PNG_PATH):
    opts = Options()
    opts.add_argument("--headless=new")          # 无头模式，不弹窗
    # opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    # opts.add_argument("--window-size=2400,1800")

    service = Service(executable_path='/home/ivan/chromedriver-linux64-145.0.7632.117/chromedriver')
    # service = Service(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service, options=opts)
    driver.get(f'file://{os.path.abspath(HTML_PATH)}')
    time.sleep(3)
    driver.set_window_size(2400, 1800)  # 可选：设置窗口大小
    time.sleep(2)
    driver.save_screenshot(PNG_PATH)
    driver.quit()

def vis_save_map(sensor_loc_file_path, file_name, id_lst):
    df=pd.read_csv(sensor_loc_file_path, names=['station_id','lat','lon'])
    mean_lat=df['lat'].mean()
    mean_lon=df['lon'].mean()
    
    # if "PEMS" in file_name:
    #     m = folium.Map(location=(mean_lat,mean_lon))
    #     sw = df[['lat', 'lon']].min().values.tolist()
    #     ne = df[['lat', 'lon']].max().values.tolist()
    #     m.fit_bounds([sw, ne])
    # else:
    m=folium.Map(location=(mean_lat,mean_lon),zoom_start=13)
    
    color = "#2ca02c"
    for i in df.iterrows():
        #print(i)
        tmp_lat=i[1]['lat']
        tmp_lon=i[1]['lon']
        tmp_id=i[1]['station_id']
        #print(tmp_lat,tmp_lon)
        
        folium.Circle(
            location=(tmp_lat,tmp_lon), 
            radius=50,
            color=f'{color}',
            fill=True,
            fill_color=f'light{color}',
            popup=str(int(tmp_id))+' : '+str(tmp_lat)+' , '+str(tmp_lon)
        ).add_to(m)
        
    scale_color_map = {0: "#ff7f0e", 1: '#D11226', 2: '#64212D', 3: '#210D10'}
    # import ipdb; ipdb.set_trace()
    for i, _l in enumerate(id_lst): # _l 
        color_i = scale_color_map[i]
        df_scale_i = df[df['station_id'].isin(_l)]
        # import ipdb; ipdb.set_trace()
        for row in df_scale_i.iterrows():
            tmp_lat = row[1]['lat']
            tmp_lon = row[1]['lon']
            tmp_id = row[1]['station_id']
            
            folium.Circle(
                location=(tmp_lat,tmp_lon), 
                radius=(1.5*(i+1)*85),
                color=f'{color_i}',
                # fill=True,
                # fill_color=f'light{color_i}',
                popup=str(int(tmp_id))+' : '+str(tmp_lat)+' , '+str(tmp_lon)
            ).add_to(m)
        
    # return m
    folium.TileLayer('CartoDB positron', attr='© CartoDB').add_to(m)
    
    # save html file
    HTML_FILE_PATH = file_name + ".html"
    m.save(HTML_FILE_PATH)
    
    PNG_FILE_PATH = file_name + ".png"
    saveImg(HTML_FILE_PATH, PNG_FILE_PATH)

# def save_gates_img(values, store_path, dataset_name, colors=None, figsize=(6, 10)):
#     indices = np.arange(len(values))

#     fig, ax = plt.subplots(figsize=figsize)

#     bars = ax.bar(
#         indices, values,
#         width=0.75,                  # 瘦一点更适合窄图
#         color=colors,
#         edgecolor='black',
#         linewidth=1.6,
#         alpha=0.85
#     )

#     ax.set_xlabel('Scale (i)', fontsize=16, labelpad=8)
#     ax.set_ylabel('Weight', fontsize=16, labelpad=8)
#     ax.set_title(
#         f'Weight Values for Each Scale of {dataset_name}',
#         fontsize=18, fontweight='bold', pad=12
#     )

#     # x轴刻度：如果柱子太多，就隔几个显示一次，防止瘦高图挤爆
#     n = len(values)
#     step = 1 if n <= 25 else (2 if n <= 50 else (5 if n <= 120 else 10))
#     xticks = indices[::step]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([str(i) for i in xticks], fontsize=13)

#     ax.tick_params(axis='y', which='major', labelsize=13, width=1.4, length=6)
#     ax.tick_params(axis='x', which='major', width=1.4, length=6)

#     for spine in ax.spines.values():
#         spine.set_linewidth(1.6)

#     ax.grid(True, axis='y', linestyle='--', linewidth=1.0, alpha=0.6)

#     # 标注
#     val_range = (max(values) - min(values) + 1e-12)
#     dy = 0.012 * val_range
#     for bar in bars:
#         h = bar.get_height()
#         display_text = "0.0" if abs(h) < 1e-4 else f"{h:.2f}"
#         va = 'bottom' if h >= 0 else 'top'
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             h + (dy if h >= 0 else -dy),
#             display_text,
#             ha='center', va=va,
#             fontsize=12, fontweight='semibold',
#             clip_on=False
#         )

#     fig.tight_layout()
#     fig.savefig(store_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt

def save_gates_img(values, store_path, dataset_name, colors=None, placeholder_at=None):
    values = np.asarray(values)
    n = len(values)

    # 在 x 轴插入一个“空槽位”的位置（默认放在最后，多一格右边距）
    if placeholder_at is None:
        placeholder_at = n
    placeholder_at = int(np.clip(placeholder_at, 0, n))  # 保证在 [0, n] 内

    # 原始柱子位置：0..n-1
    base_x = np.arange(n)
    # 在 placeholder_at 之后的柱子整体右移 1，形成空槽位
    x = base_x + (base_x >= placeholder_at).astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        x, values,
        width=0.8,
        color=colors,
        edgecolor='black',
        linewidth=1.6,
        alpha=0.85
    )

    ax.set_xlabel('Scale (i)', fontsize=16, labelpad=8)
    ax.set_ylabel('Weight', fontsize=16, labelpad=8)
    ax.set_title(
        f'Weight Values for Each Scale of {dataset_name}',
        fontsize=18, fontweight='bold', pad=12
    )

    # ----- 关键：设置 ticks + labels，并让占位 tick 不显示 -----
    xticks = np.arange(n + 1)  # 因为多插入了 1 个占位
    xticklabels = (
        [str(i) for i in range(placeholder_at)] +
        [''] +  # 占位：不显示
        [str(i) for i in range(placeholder_at, n)]
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # 进一步：把占位 tick 的刻度线也隐藏（更“干净”）
    ticks = ax.xaxis.get_major_ticks()
    if 0 <= placeholder_at < len(ticks):
        ticks[placeholder_at].tick1line.set_visible(False)
        ticks[placeholder_at].tick2line.set_visible(False)
        ticks[placeholder_at].label1.set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=13, width=1.4, length=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    ax.grid(True, axis='y', linestyle='--', linewidth=1.0, alpha=0.6)

    # 给 x 轴留出合理边距
    ax.set_xlim(-0.6, (n + 1) - 0.4)

    # 标注数字
    v_range = float(np.nanmax(values) - np.nanmin(values) + 1e-12)
    for bar in bars:
        h = bar.get_height()
        display_text = "0.0" if abs(h) < 1e-4 else f"{h:.2f}"

        if h >= 0:
            va, dy = 'bottom', 0.01 * v_range
        else:
            va, dy = 'top', -0.01 * v_range

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            display_text,
            ha='center',
            va=va,
            fontsize=12,
            fontweight='semibold',
            clip_on=False
        )

    fig.tight_layout()
    fig.savefig(store_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



# def save_gates_img(values, store_path, dataset_name, colors=None):
#     indices = np.arange(len(values))
    
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(indices, values, width=0.8, edgecolor='black', alpha=0.7, color=colors)

#     plt.xlabel('Scale (i)')
#     plt.ylabel('Weight')
#     plt.title(f'Weight Values for Each Scale of {dataset_name}')
    
#     plt.xticks(indices)
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
#     for bar in bars:
#         height = bar.get_height()
        
#         if height < 0.0001:
#             display_text = "0.0"
#         else:
#             display_text = f"{height:.2f}"
#         plt.text(bar.get_x() + bar.get_width() / 2, height, display_text, ha='center', va='bottom', fontsize=9)

#     save_path = store_path
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

def plot_and_save_gate_heatmap(gate_sg, gate_dyg, dataset_name, DATA_DIR, f_name, b_idx, 
                               weekday, hour_begin, hour_end):
    """
    绘制并保存混合专家系统门控激活程度热力图
    
    参数:
        gate_sg (np.array): 形状为 [num_expert, ] 的蓝色门控激活数组
        gate_dyg (np.array): 形状为 [num_expert, ] 的红色门控激活数组
        dataset_name (str): 数据集名称
        b_idx (int/str): 样本批次或索引标识
        weekday (str): 星期几 (如 "Monday")
        hour_begin (str/int): 起始小时 (如 "08:00" 或 8)
        hour_end (str/int): 结束小时 (如 "09:00" 或 9)
        save_path (str): 图片保存路径 (如 "./figs/heatmaps/sample_1.png")
    """
    # 确保输入为 numpy 数组
    gate_sg = np.array(gate_sg)
    gate_dyg = np.array(gate_dyg)
    num_expert = len(gate_sg)
    
    # 拼接大标题 (可以根据你的实际需求微调格式)
    # HeadLine = f"{dataset_name} {weekday}-{hour_begin}~{hour_end}"
    HeadLine = "SD-MoE Experts Activation Heatmap on " + dataset_name + ": " + weekday + "-" + hour_begin + "~" + hour_end
    
    # ========== 配色 ==========
    blue_cmap = cm.get_cmap("Blues")
    red_cmap  = cm.get_cmap("Reds")
    
    # ========== 创建画布 ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3.5), 
                                    gridspec_kw={"hspace": 0.6})
    fig.suptitle(HeadLine, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.82, hspace=0.5) 
    
    axes  = [ax1, ax2]
    gates = [gate_sg, gate_dyg]
    titles = ["SG-MoE", "DyG-MoE"]
    cmaps  = [blue_cmap, red_cmap]
    
    # ========== 绘制热力图 ==========
    for ax, data, title, cmap_used in zip(axes, gates, titles, cmaps):
        line_width = 2.0  # 分割线的宽度
        for j in range(num_expert + 1):  # 注意是 num_expert + 1，包含首尾边界
            ax.add_patch(plt.Rectangle(
                (j - 0.5 - line_width/2, -0.5), line_width, 1, 
                color="black", 
                edgecolor="none", linewidth=0
            ))
            
        # 2. 再画实际的色块（上层，稍微缩进一点避免盖住白线）
        for j in range(num_expert):
            color = cmap_used(data[j])
            ax.add_patch(plt.Rectangle(
                (j - 0.5 + line_width/2, -0.5), 1 - line_width, 1, 
                color=color, 
                edgecolor="none", linewidth=0  # 不要边框
            ))
            
        # 3. 填数字 (统一使用黑色字体)
        for j in range(num_expert):
            ax.text(j, 0, f"{data[j]:.2f}", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")
        
        # 坐标轴设置
        ax.set_xlim(-0.5, num_expert - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(range(num_expert))
        ax.set_xticklabels([f"E{i}" for i in range(num_expert)])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, pad=6, color="#333333")
    
    # ========== 保存图片 ==========
    # 自动创建不存在的文件夹
    pic_name = f_name[:-4] + f"-{b_idx:02d}.png"
    save_path = os.path.join(DATA_DIR, pic_name)
    # folder = os.path.dirname(save_path)
    # if folder and not os.path.exists(folder):
    #     os.makedirs(folder, exist_ok=True)
        
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()  # 关闭画布，防止内存泄漏

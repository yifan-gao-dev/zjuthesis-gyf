import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import torch
import random
from argparse import ArgumentParser

from ipdb import set_trace
from tqdm import tqdm

from utils import load_pkl, load_dataset_data
from ch4_draw_tools import draw_boxplot_pic, vis_save_map
from ch4_fartherest_point_sample import EuclideanFPSSamplerTorch as Sampler

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='METR-LA')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--s', type=int, default=3)

    args = parser.parse_args()
    return args

def main(args):
    DATASET_NAME = args.data
    SAVE_DIR = os.path.join("Vis-Output", f"{DATASET_NAME}", "MultiScalesBoxPlot")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    adj_file_path = f"./{DATASET_NAME}/adj_mx.pkl"
    # import ipdb; ipdb.set_trace()
    coord_csv_path = f"./{DATASET_NAME}/graph_sensor_locations_{DATASET_NAME}.csv"
    sensor_ids_lst, sensor_adj_id_map, adj_mx = load_pkl(adj_file_path)
    
    sampler = Sampler(
        coord_csv_path=coord_csv_path,
        real2adj=sensor_adj_id_map,
        device="cuda"
    )

    T = args.seq_len
    raw_data = load_dataset_data(DATASET_NAME)[..., 0]
    TOTAL_LEN, N = raw_data.shape
    
    scales_idx_lst = []
    vis_map_scales_idx_lst = []
    for i in range(1, 3):
        scale_i_idx = sampler.sample(N //(4) ** i, init_idx=0, return_real_id=False)
        vis_map_scale_i_idx = sampler.sample(N //(4) ** i, init_idx=0, return_real_id=True)
        scales_idx_lst.append(scale_i_idx)
        vis_map_scales_idx_lst.append(vis_map_scale_i_idx)
    # set_trace()
    select_period_idx = [random.randint(0, TOTAL_LEN - 2*T) for _ in range(args.k)]
    
    scales_data = []
    for idx in tqdm(select_period_idx):
        ori_scale_data_i = raw_data[idx:idx+T, :]
        ori_scale_data_i[ori_scale_data_i == 0] = np.nan
        
        ori_scale_data_mean = np.nanmean(ori_scale_data_i, axis=0)
        ori_scale_data_mean = ori_scale_data_mean[~np.isnan(ori_scale_data_mean)]
        scales_data_lst_i = [ori_scale_data_mean]
        
        for scale_idx in scales_idx_lst:
            scale_j_data = ori_scale_data_i[:, scale_idx]
            scale_j_data_mean = np.nanmean(scale_j_data, axis=0)
            scale_j_data_mean = scale_j_data_mean[~np.isnan(scale_j_data_mean)]
            scales_data_lst_i.append(scale_j_data_mean)
        # set_trace()
        scales_data.append(scales_data_lst_i)
        BOXPLT_ID = os.path.join(SAVE_DIR, f"FPS-{DATASET_NAME}-{idx:05d}.png")
        REALMAP_ID = os.path.join(SAVE_DIR, f"FPS-REALMAP-{DATASET_NAME}-{idx:05d}")
        # set_trace()
        try:
            draw_boxplot_pic(scales_data_lst_i, BOXPLT_ID, ["#2ca02c", "#ff7f0e", "#d63344"]) # ["#2ca02c", "#ff7f0e", "#d62728"]
            # set_trace()
            # vis_save_map(coord_csv_path, REALMAP_ID, vis_map_scales_idx_lst)
        except ValueError as e:
            continue# set_trace()
    
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)

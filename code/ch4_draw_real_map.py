import os

import torch
import numpy as np
from random import randint
from utils import load_pkl

from tqdm import tqdm
from argparse import ArgumentParser

from ch4_draw_tools import vis_save_map, save_gates_img, plot_and_save_gate_heatmap

time_in_day_map = {
    0: "00:00", 1: "01:00", 2: "02:00", 3: "03:00", 4: "04:00", 5: "05:00",
    6: "06:00", 7: "07:00", 8: "08:00", 9: "09:00", 10: "10:00", 11: "11:00",
    12: "12:00", 13: "13:00", 14: "14:00", 15: "15:00", 16: "16:00", 17: "17:00",
    18: "18:00", 19: "19:00", 20: "20:00", 21: "21:00", 22: "22:00", 23: "23:00"
}

day_in_week_map = {
    0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
    4: "Thursday", 5: "Friday", 6: "Saturday"
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='METR-LA')
    parser.add_argument('--window_size', type=str, default='W4')
    
    args = parser.parse_args()
    
    DATASET_NAME = args.data
    file_path = f"./{DATASET_NAME}/adj_mx.pkl"
    if args.data == 'METR-LA' or args.data == 'PEMS-BAY':
        sensor_ids_lst, sensor_adj_id_map, adj_mx = load_pkl(file_path)
    else:
        adj_mx = load_pkl(file_path)
    # import ipdb; ipdb.set_trace()
    adj_sensor_id_map = {value : key for key, value in sensor_adj_id_map.items()}    # <sensor real id, sensor adj id>
    
    # get all .pth files' names
    WINDOW_SIZE = args.window_size
    # DATASET_SUB_NAME = f"{DATASET_NAME}-{WINDOW_SIZE}"
    DATASET_SUB_NAME = "Scale-Time-CorPEMS-BAY"
    DATASET_ROOT_DIR = f"InputData/{DATASET_SUB_NAME}"
    
    FILES_LST = os.listdir(DATASET_ROOT_DIR)
    FILES_LST = [f[:-4] for f in FILES_LST]
    DATASET_MAP = f"{DATASET_NAME}/graph_sensor_locations_{DATASET_NAME}.csv"
    
    # makedir: {DATASET}/html and {DATASET}/pics
    OUTDIR = f"Vis-Output/{DATASET_SUB_NAME}"
    os.makedirs(OUTDIR, exist_ok=True)
    
    # process every .pth file
    FILES_LST.sort()
    for i, f_name in tqdm(enumerate(FILES_LST), desc="Processing files..."):
        # new_f_prefix = f_name[:-4]
        NPZ_FILE_PATH = os.path.join(DATASET_ROOT_DIR, f_name + ".npz")
        PTH_FILE_PATH = os.path.join(DATASET_ROOT_DIR, f_name + ".pth")
        
        idx_lst = torch.load(PTH_FILE_PATH, weights_only=True)        
        B, _ = idx_lst[0].shape
        size = B // 2
        rand_batch_idx = np.random.randint(0, B-1)
        
        
        # draw expert activation
        data = np.load(NPZ_FILE_PATH)
        his_data    = data['his_data']
        gates_sg      = data['imp_sg']
        gates_dyg     = data['imp_dyg']
        scale_gates = data['scale_gates']
        hours = np.round(his_data[..., 1] * 24).astype(int)
        hours = np.where(hours == 24, 0, hours)
        
        weekdays = np.round(his_data[..., 2] * 7).astype(int)
        weekdays = np.where(weekdays == 7, 0, weekdays)
        
        # for rand_batch_idx in rand_batch_idx:
    # draw activate
        hour_ = hours[rand_batch_idx, :, 0]
        hour_begin_str = time_in_day_map[hour_[0]]
        hour_end_str = time_in_day_map[hour_[-1]]
        weekday_str = day_in_week_map[weekdays[rand_batch_idx, 0, 0]]
        # print(hour_begin_str + "~" + hour_end_str + ", " + weekday_str)
        gate_sg = gates_sg[rand_batch_idx, :]
        gate_dyg = gates_dyg[rand_batch_idx, :]
        plot_and_save_gate_heatmap(gate_sg, gate_dyg, DATASET_NAME, OUTDIR, f_name, rand_batch_idx, weekday_str, hour_begin_str, hour_end_str)
            
        # draw map
        # for rand_batch_idx in rand_batch_idx:
        idx_lst = [_l[rand_batch_idx].cpu().numpy() for _l in idx_lst]
        sensor_id_lst = []
        for _l in idx_lst:
            sensor_ids_lst_s = []
            for i in range(len(_l)):
                sensor_ids_lst_s.append(int(adj_sensor_id_map[_l[i]]))
            sensor_id_lst.append(sensor_ids_lst_s)
        
        _FILE_NAME = os.path.join(OUTDIR, f"{f_name[:-4]}-{rand_batch_idx:02d}-realmap")
        vis_save_map(DATASET_MAP, _FILE_NAME, sensor_id_lst)
            
            

    

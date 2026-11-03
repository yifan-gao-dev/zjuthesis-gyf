import csv
import math
import torch

from typing import Dict, List, Optional
from argparse import ArgumentParser

from utils import *

class EuclideanFPSSamplerTorch:
    """
    GPU-friendly Farthest Point Sampling (FPS)
    using Euclidean distance on locally projected coordinates.
    """

    def __init__(
        self,
        coord_csv_path: str,
        real2adj: Dict[str, int],
        device: str = "cuda",
        has_header: bool = False,
        id_col: int = 0,
        lat_col: int = 1,
        lon_col: int = 2,
    ):
        self.device = torch.device(device)
        self.real2adj = real2adj

        # load & align data
        self.real_ids, self.adj_ids, lats, lons = self._load_csv(
            coord_csv_path,
            has_header,
            id_col,
            lat_col,
            lon_col,
        )

        # project lat/lon to local xy (km)
        self.xy = self._latlon_to_xy(lats, lons).to(self.device)
        self.num_points = self.xy.shape[0]

    def _load_csv(
        self,
        path: str,
        has_header: bool,
        id_col: int,
        lat_col: int,
        lon_col: int,
    ):
        real_ids = []
        adj_ids = []
        lats = []
        lons = []

        with open(path, "r") as f:
            reader = csv.reader(f)
            if has_header:
                next(reader)

            for row in reader:
                real_id = str(row[id_col])
                if real_id not in self.real2adj:
                    continue

                real_ids.append(real_id)
                adj_ids.append(self.real2adj[real_id])
                lats.append(float(row[lat_col]))
                lons.append(float(row[lon_col]))

        if len(real_ids) == 0:
            raise ValueError("No valid sensor points found.")

        return real_ids, adj_ids, lats, lons

    def _latlon_to_xy(self, lats, lons):
        """
        Local equirectangular projection (km)
        """
        R = 6371.0
        lat0 = sum(lats) / len(lats)
        lon0 = sum(lons) / len(lons)

        lat = torch.tensor(lats, dtype=torch.float32)
        lon = torch.tensor(lons, dtype=torch.float32)

        lat0 = math.radians(lat0)
        lon0 = math.radians(lon0)

        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)

        x = R * math.cos(lat0) * (lon - lon0)
        y = R * (lat - lat0)

        return torch.stack([x, y], dim=1)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        init_idx: Optional[int] = None,
        return_real_id: bool = False,
    ) -> List[int]:
        """
        FPS sampling

        Parameters
        ----------
        num_samples : int
            number of sampled points
        init_idx : Optional[int]
            initial index (None -> random)
        return_real_id : bool
            return real_id instead of adj_id

        Returns
        -------
        List[int] or List[str]
        """
        if num_samples >= self.num_points:
            return self.real_ids if return_real_id else self.adj_ids

        # initialize
        if init_idx is None:
            init_idx = torch.randint(
                0, self.num_points, (1,), device=self.device
            ).item()

        selected = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        selected[0] = init_idx

        # min distance to selected set
        dist = torch.full(
            (self.num_points,),
            float("inf"),
            device=self.device,
        )

        for i in range(1, num_samples):
            last = selected[i - 1]

            diff = self.xy - self.xy[last]          # [N, 2]
            d = (diff * diff).sum(dim=1)             # squared distance

            dist = torch.minimum(dist, d)
            selected[i] = torch.argmax(dist)

        selected = selected.cpu().tolist()

        if return_real_id:
            return [int(self.real_ids[i]) for i in selected]
        else:
            return [self.adj_ids[i] for i in selected]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='METR-LA')
    parser.add_argument('--window_size', type=str, default='W4')
    
    args = parser.parse_args()
    
    DATASET_NAME = args.data
    adj_file_path = f"./{DATASET_NAME}/adj_mx.pkl"
    coord_csv_path = f"./{DATASET_NAME}/graph_sensor_locations_{DATASET_NAME}.csv"
    sensor_ids_lst, sensor_adj_id_map, adj_mx = load_pkl(adj_file_path)

    sampler = EuclideanFPSSamplerTorch(
        coord_csv_path=coord_csv_path,
        real2adj=sensor_adj_id_map,
        device="cuda",   # or "cpu"
    )

    # 采样 51 个节点（返回 adj_id）
    adj_ids_lvl1 = sampler.sample(51)

    # 采样 12 个节点（返回 real_id）
    real_ids_lvl2 = sampler.sample(12, return_real_id=True)
    print(adj_ids_lvl1)

"""
fps_207 = sampler.sample(207)
fps_51  = sampler.sample(51)
fps_12  = sampler.sample(12)

"""
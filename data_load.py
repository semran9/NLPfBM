import numpy as np
import torch
from fractional_noise import SparseGPNoise
from latent_sde import LatentSDE

def transform_vects(ys, t, dt = 5*1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    ys = torch.from_numpy(np.matrix(ys))
    ts = torch.from_numpy(np.linspace(0, t, len(ys)))
    ts, ys = ts.to(device), ys.to(device)[:, None]
    white_noise = SparseGPNoise(
        t0=ts[0], t1=ts[-1], dt=dt, num_steps=len(ts), num_inducings=100
    )
    ts = ts.float()
    return ts, ys, white_noise








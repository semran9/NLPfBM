import matplotlib.pyplot as plt
import torch
def plot_h(hurst_fn, var, ts, shift=0.0):
    """Plot Hurst function"""
    plt.figure(figsize=(5, 4))

    with torch.no_grad():
        ht = hurst_fn(ts)
        plt.plot(ts, ht, label="ours", alpha=0.8)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$h(t)$")
        plt.legend()
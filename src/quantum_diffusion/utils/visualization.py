import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_quantum_distribution(
    q: np.ndarray,
    p: np.ndarray,
    exact_dist: np.ndarray = None,
    q_grid: np.ndarray = None,
    p_grid: np.ndarray = None,
    title: str = "Quantum Phase Space Distribution",
    output_path: str = "quantum_distribution.png"
):
    """绘制相空间量子分布（2D直方图+可选精确解 contours）"""
    plt.figure(figsize=(10, 8))
    
    # 采样分布直方图
    plt.hist2d(q, p, bins=100, cmap="viridis", norm=LogNorm(), alpha=0.7)
    plt.colorbar(label="Probability Density (log scale)")
    
    # 精确解contour
    if exact_dist is not None and q_grid is not None and p_grid is not None:
        q_mesh, p_mesh = np.meshgrid(q_grid, p_grid)
        plt.contour(q_mesh, p_mesh, exact_dist.T, levels=10, colors="red", linewidths=1.5, alpha=0.8)
    
    plt.xlabel("Position $q$")
    plt.ylabel("Momentum $p$")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_real_time_evolution(
    samples_list: list,
    times: np.ndarray,
    s: float = 0.0,
    output_path: str = "real_time_evolution.png"
):
    """绘制实时动力学演化（多时间点分布对比）"""
    num_plots = min(5, len(samples_list))  # 最多显示5个时间点
    selected_idx = np.linspace(0, len(samples_list)-1, num_plots, dtype=int)
    selected_samples = [samples_list[i] for i in selected_idx]
    selected_times = [times[i] for i in selected_idx]
    
    dist_type = "Wigner" if s == 0 else "Husimi Q" if s == -1 else "Glauber P"
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    
    for i, (samples, t) in enumerate(zip(selected_samples, selected_times)):
        q = samples[:, 0, 0]
        p = samples[:, 1, 0]
        axes[i].hist2d(q, p, bins=50, cmap="viridis", norm=LogNorm())
        axes[i].set_title(f"{dist_type} (t={t:.2f})")
        axes[i].set_xlabel("$q$")
        axes[i].set_ylabel("$p$")
        axes[i].axis("equal")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
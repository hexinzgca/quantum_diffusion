import numpy as np
from scipy.ndimage import gaussian_filter

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL散度：D_KL(p||q)（p为精确解，q为采样/预测分布）"""
    # 归一化
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    # 平滑避免log(0)
    p = gaussian_filter(p, sigma=1) + eps
    q = gaussian_filter(q, sigma=1) + eps
    # 计算KL散度
    return np.sum(p * np.log(p / q))

def mse_error(exact: np.ndarray, approx: np.ndarray) -> float:
    """均方误差"""
    # 归一化
    exact = exact / np.max(exact)
    approx = approx / np.max(approx)
    return np.mean((exact - approx)**2)
    
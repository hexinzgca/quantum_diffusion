import torch
import torch.nn as nn

class Base1DQuantumSystem(nn.Module):
    """一维量子系统基类（通用接口）
    所有量子系统需继承此类并实现核心方法
    """
    def __init__(self, s: float = 0.0, dissipation: dict = None, hbar: float = 1.0, device: str = "cuda"):
        super().__init__()
        self.s = s  # s-序参数
        self.dissipation = dissipation or {}  # 耗散配置
        self.hbar = hbar  # 约化普朗克常数
        self.device = device  # 计算设备

    def classical_hamiltonian(self, alpha: torch.Tensor) -> torch.Tensor:
        """经典哈密顿量（相空间α表示）
        alpha: (batch_size,) 复数张量
        返回: (batch_size,) 实数值哈密顿量
        """
        raise NotImplementedError

    def quantum_hamiltonian(self, alpha: torch.Tensor) -> torch.Tensor:
        """量子哈密顿量的s-序符号"""
        from core.quantum_operators import SOrderedProbability
        s_order = SOrderedProbability(s=self.s, hbar=self.hbar)
        H_classical = self.classical_hamiltonian(alpha)
        return s_order.weyl_symbol(alpha, H_classical)

    def dissipation_terms(self, alpha: torch.Tensor, t: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """耗散项：漂移项D1 + 扩散矩阵D2
        返回:
            D1: (batch_size,) 复数漂移项
            D2: (batch_size, 1, 1) 实对称扩散矩阵（正半定）
        """
        raise NotImplementedError

    def to(self, device: str):
        """设备迁移"""
        self.device = device
        return super().to(device)
import torch
import torch.nn as nn
from .quantum_operators import SOrderedProbability

class FokkerPlanck(nn.Module):
    """通用Fokker-Planck方程（兼容任意一维量子系统+耗散机制）
    核心：将GKSL方程映射为相空间概率分布的演化方程
    """
    def __init__(self, system: "Base1DQuantumSystem"):
        super().__init__()
        self.system = system  # 量子系统（需实现classical_hamiltonian、dissipation_terms）
        self.s_order = SOrderedProbability(s=system.s, hbar=system.hbar)

    def forward(self, W: torch.Tensor, alpha: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """计算Fokker-Planck方程的演化率dW/dt
        W: 准概率分布 (batch_size,)
        alpha: 相空间坐标 (batch_size,) 复数张量
        t: 时间（时变耗散时使用）
        """
        # 1. 系统哈密顿量项（量子动力学）
        H_classical = self.system.classical_hamiltonian(alpha)
        H_quantum = self.s_order.weyl_symbol(alpha, H_classical)
        moyal_term = self.s_order.moyal_bracket(H_quantum, W, alpha)
        ham_term = -self.s_order.hbar * 1j * moyal_term

        # 2. 耗散项（漂移+扩散）
        D1, D2 = self.system.dissipation_terms(alpha, t)  # D1:漂移 (batch_size,), D2:扩散 (batch_size,1,1)
        
        # 漂移项贡献（一阶导数）
        drift_grad = torch.autograd.grad(
            (D1 * W).sum(), alpha, create_graph=True)[0] if W.requires_grad else torch.zeros_like(alpha)
        drift_term = -drift_grad.real  # 实部保证概率守恒

        # 扩散项贡献（二阶导数）
        D2 = D2.squeeze(-1).squeeze(-1)  # (batch_size,)
        diff_grad1 = torch.autograd.grad(
            (D2 * W).sum(), alpha, create_graph=True)[0] if W.requires_grad else torch.zeros_like(alpha)
        diff_grad2 = torch.autograd.grad(
            diff_grad1.sum(), alpha.conj(), create_graph=True)[0] if diff_grad1.requires_grad else torch.zeros_like(alpha)
        diff_term = 0.5 * diff_grad2.real

        # 总演化率（实数值，概率分布守恒）
        return (ham_term + drift_term + diff_term).real

    def to(self, device: str):
        """设备迁移（重载以保证系统参数同步）"""
        super().to(device)
        self.system.to(device)
        return self
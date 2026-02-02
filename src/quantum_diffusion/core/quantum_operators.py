import numpy as np
import torch
import torch.autograd as autograd

class SOrderedProbability:
    """s-序准概率分布（支持P/W/Q分布切换）
    核心特性：实现通用s-序星积、Moyal括号、Weyl符号映射
    s=1: Glauber-Sudarshan P分布（正规序）
    s=0: Wigner分布（对称序/Weyl序）
    s=-1: Husimi Q分布（反正规序）
    """
    def __init__(self, s: float = 0.0, hbar: float = 1.0):
        self.s = s  # 序化参数（-1 ≤ s ≤ 1）
        self.hbar = hbar  # 约化普朗克常数

    def weyl_symbol(self, alpha: torch.Tensor, H_classical: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量的s-序Weyl符号（通用形式）"""
        if not isinstance(alpha, torch.Tensor) or not isinstance(H_classical, torch.Tensor):
            raise TypeError("alpha和H_classical必须是torch.Tensor类型")
        
        if self.s == 0:
            # W分布：对称序，直接返回经典哈密顿量
            return H_classical
        elif self.s == -1:
            # Q分布：反正规序，高斯平滑（对应相干态平均）
            sigma = self.hbar / 2.0
            return H_classical * torch.exp(-sigma * (alpha.abs() ** 2))
        elif self.s == 1:
            # P分布：正规序，逆高斯平滑
            sigma = self.hbar / 2.0
            return H_classical * torch.exp(sigma * (alpha.abs() ** 2))
        else:
            raise ValueError(f"不支持的s-序参数：{self.s}（仅支持1/0/-1）")

    def star_product(self, A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """一维系统的s-序星积（基于Moyal乘积扩展，支持任意s）
        星积定义：A★B = A*B + (iħ/2)*(∂A/∂α·∂B/∂α* - ∂A/∂α*·∂B/∂α) + ...
        """
        # 计算各阶偏导数（创建计算图以支持高阶微分）
        A_alpha = autograd.grad(A.sum(), alpha, create_graph=True)[0] if A.requires_grad else torch.zeros_like(alpha)
        A_alpha_star = autograd.grad(A.sum(), alpha.conj(), create_graph=True)[0] if A.requires_grad else torch.zeros_like(alpha)
        B_alpha = autograd.grad(B.sum(), alpha, create_graph=True)[0] if B.requires_grad else torch.zeros_like(alpha)
        B_alpha_star = autograd.grad(B.sum(), alpha.conj(), create_graph=True)[0] if B.requires_grad else torch.zeros_like(alpha)
        
        # s-序修正项（融合不同序化的非局域效应）
        s_correction = 0.5 * self.hbar * 1j * (
            (1 + self.s) * A_alpha * B_alpha_star - 
            (1 - self.s) * A_alpha_star * B_alpha
        )
        return A * B + s_correction

    def moyal_bracket(self, A: torch.Tensor, B: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Moyal括号：{A,B} = (A★B - B★A)/(iħ)，描述量子泊松括号"""
        star_AB = self.star_product(A, B, alpha)
        star_BA = self.star_product(B, A, alpha)
        return (star_AB - star_BA) / (1j * self.hbar)

    def quasiprobability_dist(self, alpha: torch.Tensor, psi: torch.Tensor = None, n: int = 0) -> torch.Tensor:
        """生成已知量子态的s-序准概率分布（支持相干态/数态）
        psi: 波函数（可选），n: 量子数（数态时使用）
        """
        if psi is not None:
            # 从波函数计算（位置表象）
            q = torch.real(alpha) * np.sqrt(2 * self.hbar)
            p = torch.imag(alpha) * np.sqrt(2 * self.hbar)
            return self._wigner_from_wavefunction(q, p, psi)
        else:
            # 数态的解析表达式
            if self.s == 0:
                # Wigner函数（数态）
                q = torch.real(alpha) * np.sqrt(2 * self.hbar)
                p = torch.imag(alpha) * np.sqrt(2 * self.hbar)
                arg = (q**2 + p**2) / self.hbar
                hermite = torch.special.hermite_polynomial(n, 2*arg)
                return (1 / (np.pi * self.hbar)) * torch.exp(-arg) * hermite
            elif self.s == -1:
                # Q分布（数态：相干态平均）
                q = torch.real(alpha) * np.sqrt(2 * self.hbar)
                p = torch.imag(alpha) * np.sqrt(2 * self.hbar)
                wigner = self.quasiprobability_dist(alpha, n=n)
                sigma = self.hbar / 2.0
                return wigner * torch.exp(-sigma * (q**2 + p**2))
            elif self.s == 1:
                # P分布（数态：Dirac delta函数近似）
                alpha_n = torch.sqrt(torch.tensor(n, dtype=alpha.dtype, device=alpha.device))
                return torch.pi * torch.exp(-2 * torch.abs(alpha - alpha_n)**2)

    def _wigner_from_wavefunction(self, q: torch.Tensor, p: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """从位置表象波函数计算Wigner函数（基础实现）"""
        q_grid, p_grid = torch.meshgrid(q, p, indexing="ij")
        wigner = torch.zeros_like(q_grid)
        for i in range(q.shape[0]):
            for j in range(p.shape[0]):
                q_avg = q_grid[i, j]
                integral = torch.sum(
                    psi * torch.conj(psi) * torch.exp(-2j * p_grid[i, j] * q / self.hbar)
                )
                wigner[i, j] = integral.real / np.pi
        return wigner
        
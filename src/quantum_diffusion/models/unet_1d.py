import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):
    """一维U-Net（适配相空间坐标：实部+虚部→2维输入）
    核心设计：处理相空间低维输入，捕捉量子非局域关联
    """
    def __init__(
        self,
        in_channels: int = 2,  # 输入：Re(α), Im(α)
        out_channels: int = 2,  # 输出：噪声预测（实部+虚部）
        hidden_channels: int = 64,
        num_blocks: int = 3,
        time_emb_dim: int = 16
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        prev_channels = in_channels
        for i in range(num_blocks):
            self.down_blocks.append(nn.Sequential(
                nn.Conv1d(prev_channels + time_emb_dim, hidden_channels * (2**i), 3, padding=1),
                nn.GroupNorm(4, hidden_channels * (2**i)),
                nn.SiLU(),
                nn.Conv1d(hidden_channels * (2**i), hidden_channels * (2**i), 3, padding=1),
                nn.GroupNorm(4, hidden_channels * (2**i)),
                nn.SiLU()
            ))
            prev_channels = hidden_channels * (2**i)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv1d(prev_channels + time_emb_dim, hidden_channels * (2**num_blocks), 3, padding=1),
            nn.GroupNorm(4, hidden_channels * (2**num_blocks)),
            nn.SiLU(),
            nn.Conv1d(hidden_channels * (2**num_blocks), hidden_channels * (2**num_blocks), 3, padding=1),
            nn.GroupNorm(4, hidden_channels * (2**num_blocks)),
            nn.SiLU()
        )

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_blocks)):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(
                    hidden_channels * (2**(i+1)) + time_emb_dim,
                    hidden_channels * (2**i),
                    2,
                    stride=2,
                    padding=0
                ),
                nn.GroupNorm(4, hidden_channels * (2**i)),
                nn.SiLU(),
                nn.Conv1d(hidden_channels * (2**i), hidden_channels * (2**i), 3, padding=1),
                nn.GroupNorm(4, hidden_channels * (2**i)),
                nn.SiLU()
            ))

        # 输出层
        self.out_conv = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: 输入张量 (batch_size, in_channels, seq_len) → 这里seq_len=1（单点相空间坐标）
        t: 时间/虚时张量 (batch_size, 1)
        """
        # 时间嵌入
        time_emb = self.time_emb(t).unsqueeze(-1)  # (batch_size, time_emb_dim, 1)
        time_emb = time_emb.expand(-1, -1, x.shape[-1])  # 广播到序列长度

        # 下采样
        residuals = []
        h = x
        for block in self.down_blocks:
            # 拼接时间嵌入
            h = torch.cat([h, time_emb], dim=1)
            h = block(h)
            residuals.append(h)
            h = F.avg_pool1d(h, 2)  # 下采样

        # 瓶颈层
        h = torch.cat([h, time_emb], dim=1)
        h = self.bottleneck(h)

        # 上采样
        for block, res in zip(self.up_blocks, reversed(residuals)):
            h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)  # 上采样
            # 拼接残差和时间嵌入
            h = torch.cat([h, res, time_emb], dim=1)
            h = block(h)

        # 输出噪声预测
        return self.out_conv(h)
"""谐振子实时动力学测试脚本
流程：训练模型 → 模拟演化 → 可视化结果
"""
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加src目录到Python路径（核心修改）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from train.train_real_time import train_real_time_diffuser
from simulate.simulate_real_time import simulate_real_time_evolution

if __name__ == "__main__":
    # 创建模型/结果目录（确保存在）
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. 训练实时动力学模型
    train_real_time_diffuser(
        num_epochs=5000,
        batch_size=256,
        lr=1e-4,
        m=1.0,
        omega=1.0,
        s=0.0,  # Wigner分布
        dissipation={
            "type": "amplitude_damping",
            "gamma": 0.1,
            "non_markovian": False
        },
        save_path="models/harmonic_oscillator_real_time.pth"
    )

    # 2. 模拟实时演化
    simulate_real_time_evolution(
        model_path="models/harmonic_oscillator_real_time.pth",
        num_samples=1000,
        t_max=10.0,
        num_time_steps=20,
        output_path="results/harmonic_oscillator_real_time_evolution.png"
    )

    print("谐振子实时动力学测试完成！")
    print("模型保存：models/harmonic_oscillator_real_time.pth")
    print("演化结果：results/harmonic_oscillator_real_time_evolution.png")

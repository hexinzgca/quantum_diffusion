"""谐振子虚时采样测试脚本
流程：训练模型 → 生成采样 → 精度评估 → 可视化
"""
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加src目录到Python路径（核心修改）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


from train.train_imaginary_time import train_imaginary_time_diffuser
from simulate.simulate_imaginary_time import simulate_imaginary_time_sampling

if __name__ == "__main__":
    # 创建模型/结果目录（确保存在）
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. 训练虚时采样模型
    train_imaginary_time_diffuser(
        num_epochs=5000,
        batch_size=256,
        lr=1e-4,
        m=1.0,
        omega=1.0,
        s=-1.0,  # Husimi Q分布
        dissipation={
            "type": "amplitude_damping",
            "gamma": 0.05
        },
        save_path="models/harmonic_oscillator_imaginary_time.pth"
    )

    # 2. 虚时采样与评估
    simulate_imaginary_time_sampling(
        model_path="models/harmonic_oscillator_imaginary_time.pth",
        num_samples=10000,
        n_quantum=0,  # 基态
        output_path="results/harmonic_oscillator_imaginary_time_sampling.png"
    )

    print("谐振子虚时采样测试完成！")
    print("模型保存：models/harmonic_oscillator_imaginary_time.pth")
    print("采样结果：results/harmonic_oscillator_imaginary_time_sampling.png")

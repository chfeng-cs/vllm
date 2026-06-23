"""
Lab 0: 量化基础原理
===================
目标：
  1. 理解对称 / 非对称量化的数学推导
  2. 直观感受量化误差的来源
  3. 理解 per-tensor / per-channel / per-group 粒度的差异
  4. 动手实现 INT8 量化-反量化并测量误差

核心公式：
  量化: q = round(x / scale) + zero_point
  反量化: x̂  = (q - zero_point) * scale
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 0-A  基础量化函数
# ─────────────────────────────────────────────────────────────

def symmetric_quantize(x: torch.Tensor, bits: int = 8, dim=None):
    """对称量化（zero_point=0），scale 由绝对最大值决定。

    适用场景：权重量化（权重通常近似对称分布）
    """
    qmax = 2 ** (bits - 1) - 1          # INT8 → 127
    if dim is None:
        amax = x.abs().max()
    else:
        amax = x.abs().amax(dim=dim, keepdim=True)

    scale = amax / qmax
    scale = scale.clamp(min=1e-8)       # 避免除零

    q = (x / scale).round().clamp(-qmax - 1, qmax)   # INT8: [-128, 127]
    x_hat = q * scale                                  # 反量化
    return q, scale, x_hat


def asymmetric_quantize(x: torch.Tensor, bits: int = 8, dim=None):
    """非对称量化，scale + zero_point 均需存储。

    适用场景：激活量化（ReLU 后全正，zero_point 可节省表示范围）
    """
    qmin, qmax = 0, 2 ** bits - 1       # UINT8: [0, 255]
    if dim is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin = x.amin(dim=dim, keepdim=True)
        xmax = x.amax(dim=dim, keepdim=True)

    scale = (xmax - xmin) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = (qmin - xmin / scale).round().clamp(qmin, qmax)

    q = ((x / scale) + zero_point).round().clamp(qmin, qmax)
    x_hat = (q - zero_point) * scale
    return q, scale, zero_point, x_hat


# ─────────────────────────────────────────────────────────────
# 0-B  量化粒度实验
# ─────────────────────────────────────────────────────────────

def quant_granularity_experiment():
    """对比 per-tensor / per-channel / per-group 的量化误差。

    核心结论：
      粒度越细 → scale 精度越高 → 量化误差越小 → 但需要存储更多 scale
    """
    torch.manual_seed(42)
    # 模拟一个线性层权重 [out_features=64, in_features=128]
    W = torch.randn(64, 128)
    # 给部分 channel 注入大幅偏移，模拟真实权重分布不均
    W[0] *= 10.0
    W[32] *= 0.1

    def snr(original, reconstructed):
        signal = original.pow(2).mean()
        noise = (original - reconstructed).pow(2).mean()
        return 10 * torch.log10(signal / (noise + 1e-12)).item()

    # Per-tensor
    _, _, W_pt = symmetric_quantize(W, bits=8, dim=None)
    # Per-channel (out_channel 维度，每行一个 scale)
    _, _, W_pc = symmetric_quantize(W, bits=8, dim=1)
    # Per-group (group_size=32)
    group_size = 32
    W_pg = W.reshape(64, -1, group_size)
    _, _, W_pg_hat = symmetric_quantize(W_pg, bits=8, dim=2)
    W_pg_hat = W_pg_hat.reshape_as(W)

    print("=" * 55)
    print("量化粒度对比（SNR dB，越高越好）")
    print("=" * 55)
    print(f"  Per-tensor  (1 个 scale):          {snr(W, W_pt):.2f} dB")
    print(f"  Per-channel (64 个 scale):         {snr(W, W_pc):.2f} dB")
    print(f"  Per-group32 ({64*128//group_size} 个 scale): {snr(W, W_pg_hat):.2f} dB")
    print()
    print("关键洞察：per-channel > per-tensor，per-group 在 group_size≤128 时")
    print("  几乎等价 per-column，精度接近 FP16 但存储开销更大。")


# ─────────────────────────────────────────────────────────────
# 0-C  量化误差来源：outlier 分析
# ─────────────────────────────────────────────────────────────

def outlier_analysis():
    """演示激活中的 outlier 是量化误差的主要来源。

    LLM 激活的 outlier 现象（Dettmers et al. 2022 LLM.int8()）：
      - 某些 hidden dim 的激活值比均值大 10-100x
      - 这些 outlier 强制 scale 变大，导致正常激活精度损失
    """
    torch.manual_seed(0)
    seq_len, hidden = 32, 512
    act = torch.randn(seq_len, hidden)

    # 注入 outlier（模拟 LLM 激活中的 emergent feature）
    act[:, [6, 42, 137, 299, 486]] *= 60.0

    _, _, act_hat_pt = symmetric_quantize(act, bits=8, dim=None)     # per-tensor
    _, _, act_hat_pc = symmetric_quantize(act, bits=8, dim=1)        # per-token

    err_pt = (act - act_hat_pt).abs().mean().item()
    err_pc = (act - act_hat_pc).abs().mean().item()

    print("=" * 55)
    print("Outlier 对激活量化的影响")
    print("=" * 55)
    print(f"  激活绝对值均值:     {act.abs().mean():.4f}")
    print(f"  outlier 列绝对值均值: {act[:, 6].abs().mean():.4f}")
    print(f"  Per-tensor  MAE:    {err_pt:.6f}")
    print(f"  Per-token   MAE:    {err_pc:.6f}")
    print()
    print("结论：per-token 量化把 scale 限制在每个 token 内部，")
    print("  大幅降低 outlier 跨 token 的污染效应。")
    print("  这正是 SmoothQuant / AWQ 要解决的核心问题。")


# ─────────────────────────────────────────────────────────────
# 0-D  数值格式对比：INT4 vs INT8 vs FP8
# ─────────────────────────────────────────────────────────────

def numeric_format_comparison():
    """对比 INT4 / INT8 / FP8(E4M3) 的表示范围与精度。

    FP8 E4M3 格式（torch.float8_e4m3fn）：
      - 指数位 4 bit，尾数 3 bit
      - 表示范围：[-448, 448]（比 INT8 范围更大）
      - 但精度是非均匀的（浮点的本质）
    """
    print("=" * 55)
    print("数值格式对比")
    print("=" * 55)

    formats = {
        "INT4  (symmetric)": {"bits": 4,  "qmin": -8,   "qmax": 7,   "type": "int"},
        "INT8  (symmetric)": {"bits": 8,  "qmin": -128, "qmax": 127, "type": "int"},
        "FP8 E4M3":          {"bits": 8,  "qmin": None, "qmax": None, "type": "fp8"},
    }

    for name, fmt in formats.items():
        if fmt["type"] == "int":
            n_values = 2 ** fmt["bits"]
            print(f"  {name}: {n_values} 个离散值, 范围 [{fmt['qmin']}, {fmt['qmax']}]")
        else:
            # FP8 E4M3: max normal = 448
            try:
                info = torch.finfo(torch.float8_e4m3fn)
                print(f"  {name}: range [{info.min:.1f}, {info.max:.1f}], "
                      f"eps={info.eps:.6f}")
            except AttributeError:
                print(f"  {name}: range [-448, 448] (需要 PyTorch >= 2.1)")

    print()
    print("INT4 vs INT8 内存节省：2x，精度损失约 1-2 ppl（LLM 场景）")
    print("FP8 vs BF16 内存节省：2x，精度几乎无损（H100 原生支持）")


# ─────────────────────────────────────────────────────────────
# 0-E  W4A16 vs W8A8 推理分析
# ─────────────────────────────────────────────────────────────

def inference_mode_analysis():
    """分析不同量化推理模式的计算和内存特征。

    W[bits]A[bits] 命名约定：
      W4A16 = 权重 INT4，激活 FP16（反量化后做 FP16 GEMM）
      W8A8  = 权重 INT8，激活 INT8（直接做 INT8 GEMM）
      W4A8  = 权重 INT4，激活 INT8（混合精度）

    带宽 vs 算力：
      - Decode 阶段（batch=1）：memory-bound，W4A16 占优（减少权重带宽）
      - Prefill 阶段（长序列）：compute-bound，W8A8 占优（INT8 GEMM 吞吐更高）
    """
    print("=" * 55)
    print("量化推理模式分析")
    print("=" * 55)

    configs = [
        ("FP16  W16A16", 16, 16, 1.0,  1.0),
        ("INT8  W8A8",   8,  8,  0.5,  2.0),   # 内存 0.5x，GEMM 吞吐 2x
        ("INT4  W4A16",  4,  16, 0.25, 1.0),   # 只节省权重内存
        ("INT4  W4A8",   4,  8,  0.25, 2.0),
    ]

    print(f"  {'方案':<16} {'权重bits':>8} {'激活bits':>8} "
          f"{'权重内存':>10} {'GEMM加速':>10}")
    print("  " + "-" * 54)
    for name, wb, ab, mem_ratio, gemm_ratio in configs:
        print(f"  {name:<16} {wb:>8} {ab:>8} "
              f"{mem_ratio:>9.2f}x {gemm_ratio:>9.2f}x")

    print()
    print("Decode 瓶颈 = 显存带宽 → W4A16 优先")
    print("Prefill 瓶颈 = 计算量  → W8A8  优先（INT8 Tensor Core）")


if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  Lab 0: 量化基础原理")
    print("█" * 55 + "\n")

    quant_granularity_experiment()
    print()
    outlier_analysis()
    print()
    numeric_format_comparison()
    print()
    inference_mode_analysis()

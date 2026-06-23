"""
Lab 3: SmoothQuant — 平滑量化难度从激活迁移到权重
==================================================
论文：SmoothQuant: Accurate and Efficient Post-Training Quantization
      for Large Language Models
      Xiao et al., 2022  https://arxiv.org/abs/2211.10438

核心问题：
  激活的量化比权重更难！
  原因：LLM 激活中存在 outlier（某些 channel 值比均值大 100x）

  典型激活分布：
    普通 channel：范围 [-1, 1]
    outlier channel：范围 [-100, 100]
    → per-tensor INT8 量化会严重损失普通 channel 的精度

核心思想（数学等价变换）：
  对于 Y = X @ W.T，引入 per-channel 缩放因子 s：
    Y = (X / s_diag) @ (s_diag @ W.T)
    Y = X_smooth @ W_smooth.T

  其中：
    s_i = max|X_i|^α / max|W_i|^(1-α)    (α ∈ [0,1] 迁移强度)
    X_smooth = X / s    (激活 outlier 被平滑)
    W_smooth = W * s    (权重吸收了激活的 outlier)

  α=0：不迁移（纯激活量化）
  α=0.5：平衡迁移（默认，推荐）
  α=1.0：完全迁移到权重（等价 AWQ 思路）

关键优势（相比 GPTQ/AWQ）：
  SmoothQuant 的 s 可以融入 LayerNorm 的 scale：
    LayerNorm_output / s → 下一层 Linear 输入已 smooth
  → 推理时零额外开销（不需要运行时除法）

vLLM 相关：
  SmoothQuant 产出的 checkpoint 通常用 W8A8 格式，
  vLLM 通过 compressed-tensors 格式加载
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 3-A  SmoothQuant 等价变换演示
# ─────────────────────────────────────────────────────────────

def smooth_transform_demo():
    """演示 SmoothQuant 等价变换消除激活 outlier。"""
    torch.manual_seed(42)
    batch, seq_len, hidden = 4, 16, 128

    # 构造有 outlier 的激活（模拟 LLM 中间层激活）
    X = torch.randn(batch * seq_len, hidden) * 0.5
    outlier_channels = [5, 23, 67, 99]
    X[:, outlier_channels] *= 80.0   # 模拟 emergent outlier

    # 权重（相对均匀）
    W = torch.randn(64, hidden) * 0.02

    def int8_quant(x: torch.Tensor, per_token: bool = True) -> tuple:
        """INT8 对称量化，返回 (量化值, scale, 反量化值)"""
        if per_token:
            scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
        else:
            scale = x.abs().max() / 127.0
        q = (x / scale.clamp(1e-8)).round().clamp(-128, 127)
        return q, scale, q * scale

    def measure_snr(original, reconstructed):
        signal = original.pow(2).mean()
        noise = (original - reconstructed).pow(2).mean()
        return (10 * torch.log10(signal / (noise + 1e-12))).item()

    print("=" * 60)
    print("SmoothQuant 等价变换效果")
    print("=" * 60)

    # 直接量化激活（per-tensor）
    _, _, X_q_pt = int8_quant(X, per_token=False)
    snr_naive = measure_snr(X, X_q_pt)

    # per-token 量化激活
    _, _, X_q_tok = int8_quant(X, per_token=True)
    snr_token = measure_snr(X, X_q_tok)

    # SmoothQuant（alpha=0.5）
    alpha = 0.5
    x_max = X.abs().amax(dim=0)       # [hidden]：每个通道激活的最大值
    w_max = W.abs().amax(dim=0)       # [hidden]：权重对应 in_channel 的最大值

    s = x_max.pow(alpha) / w_max.pow(1 - alpha).clamp(1e-8)
    s = s.clamp(1e-8)

    X_smooth = X / s.unsqueeze(0)     # [batch*seq, hidden]
    W_smooth = W * s.unsqueeze(0)     # [out, hidden]

    _, _, X_smooth_q = int8_quant(X_smooth, per_token=False)
    snr_smooth = measure_snr(X_smooth, X_smooth_q)

    print(f"\n  激活中 outlier 通道的绝对均值：{X[:, outlier_channels[0]].abs().mean():.2f}")
    print(f"  激活中普通通道的绝对均值：      {X[:, 0].abs().mean():.2f}")
    print()
    print(f"  INT8 Per-tensor 量化 SNR：{snr_naive:.2f} dB")
    print(f"  INT8 Per-token  量化 SNR：{snr_token:.2f} dB")
    print(f"  INT8 SmoothQuant(α=0.5) SNR：{snr_smooth:.2f} dB")
    print()

    # 验证输出等价性
    Y_fp    = X @ W.T
    Y_smooth = X_smooth_q @ (W_smooth / s.unsqueeze(0)).T
    err_output = (Y_fp - Y_smooth).abs().mean().item()
    print(f"  输出误差（SmoothQuant）：{err_output:.6f}")
    print()

    # 可视化 outlier 被消除
    x_max_after = X_smooth.abs().amax(dim=0)
    ratio = (x_max / x_max_after).mean().item()
    print(f"  Smooth 后 outlier 被压缩比：{ratio:.2f}x")
    print(f"  SmoothQuant 把量化难度从激活迁移到权重，")
    print('  使得激活的分布更"平坦"，per-tensor INT8 精度大幅提升。')


# ─────────────────────────────────────────────────────────────
# 3-B  Alpha 参数敏感性分析
# ─────────────────────────────────────────────────────────────

def alpha_sensitivity():
    """分析 alpha 参数对量化误差的影响。"""
    torch.manual_seed(0)
    n, hidden, out = 64, 256, 128

    X = torch.randn(n, hidden)
    X[:, [10, 50, 150, 200]] *= 60.0
    W = torch.randn(out, hidden) * 0.02

    x_max = X.abs().amax(dim=0)
    w_max = W.abs().amax(dim=0).clamp(1e-8)
    Y_fp = X @ W.T

    qmax = 127

    print("=" * 60)
    print("Alpha 参数敏感性分析")
    print("=" * 60)
    print(f"  {'Alpha':>6} {'激活SNR':>10} {'权重SNR':>10} {'输出MSE':>12}")
    print("  " + "-" * 42)

    for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(1e-8)

        X_s = X / s
        W_s = W * s

        # 量化激活（per-tensor）
        scale_x = X_s.abs().max() / qmax
        X_q = (X_s / scale_x.clamp(1e-8)).round().clamp(-128, qmax) * scale_x

        # 量化权重（per-column）
        scale_w = W_s.abs().amax(dim=1, keepdim=True) / qmax
        W_q = (W_s / scale_w.clamp(1e-8)).round().clamp(-128, qmax) * scale_w

        def snr(a, b):
            return (10 * torch.log10(
                a.pow(2).mean() / (a-b).pow(2).mean().clamp(1e-12)
            )).item()

        snr_x = snr(X_s, X_q)
        snr_w = snr(W_s, W_q)

        # 推理：反量化后还原
        W_q_restored = W_q / s
        Y_q = X_q @ W_q_restored.T
        mse = F.mse_loss(Y_q, Y_fp).item()

        print(f"  {alpha:>6.1f} {snr_x:>10.2f} {snr_w:>10.2f} {mse:>12.6f}")

    print()
    print("  观察：α=0.5 通常在激活和权重量化误差间取得最佳平衡。")
    print("  α 越大 → 越多难度迁移到权重 → 权重 SNR 下降但激活 SNR 上升。")


# ─────────────────────────────────────────────────────────────
# 3-C  融入 LayerNorm 的零开销实现
# ─────────────────────────────────────────────────────────────

class SmoothLayerNorm(nn.Module):
    """LayerNorm 融合 SmoothQuant scale，推理时无额外开销。

    原理：
      LayerNorm output = (x - mean) / std * γ + β
      SmoothQuant 后：(LayerNorm output) / s
      合并：weight_new = γ / s, bias_new = β / s
    """

    def __init__(self, normalized_shape: int, smooth_scale: torch.Tensor):
        super().__init__()
        self.normalized_shape = normalized_shape
        gamma = torch.ones(normalized_shape)
        beta = torch.zeros(normalized_shape)
        s = smooth_scale.clamp(1e-8)
        # 融合 smooth_scale 进 LayerNorm 参数
        self.weight = nn.Parameter(gamma / s)
        self.bias = nn.Parameter(beta / s)

    def forward(self, x):
        return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias)


def demonstrate_fused_layernorm():
    """演示 smooth scale 融入 LayerNorm 实现零开销。"""
    torch.manual_seed(0)
    batch, seq, hidden = 2, 8, 64

    x = torch.randn(batch, seq, hidden)
    x[:, :, [3, 15, 47]] *= 30.0

    s = x.reshape(-1, hidden).abs().amax(dim=0).pow(0.5)
    s = (s / s.mean()).clamp(0.1, 10.0)

    # 方法 1：LayerNorm + 手动除以 s（推理时需要额外运算）
    ln = nn.LayerNorm(hidden)
    with torch.no_grad():
        out1 = ln(x) / s.unsqueeze(0).unsqueeze(0)

    # 方法 2：融合 LayerNorm（零额外开销）
    smooth_ln = SmoothLayerNorm(hidden, s)
    with torch.no_grad():
        out2 = smooth_ln(x)

    diff = (out1 - out2).abs().max().item()
    print("=" * 60)
    print("SmoothQuant 融入 LayerNorm 验证")
    print("=" * 60)
    print(f"  LN + /s  vs  SmoothLN 最大差异：{diff:.2e}（应 ≈ 0）")
    print()
    print("  在实际推理中，smooth scale 被 offline 融入 LayerNorm 的 γ/β，")
    print("  推理时只需执行标准 LayerNorm，无需额外除法操作。")
    print("  这是 SmoothQuant 相比 AWQ 的工程优势。")


# ─────────────────────────────────────────────────────────────
# 3-D  SmoothQuant vs GPTQ vs AWQ 三方对比
# ─────────────────────────────────────────────────────────────

def three_way_comparison():
    print("=" * 60)
    print("SmoothQuant vs AWQ vs GPTQ 三方对比")
    print("=" * 60)
    rows = [
        ("目标精度",   "W8A8",          "W4A16",         "W4A16 / W4A8"),
        ("激活量化",   "是（INT8）",     "否",             "部分（W4A8）"),
        ("校正数据",   "~512 samples",  "~128 samples",  "~128 samples"),
        ("量化时间",   "分钟级",         "分钟级",         "小时级（大模型）"),
        ("推理加速",   "2x（INT8 GEMM)","~3.5x decode",  "~3.5x decode"),
        ("精度损失",   "极小",           "小",            "最小"),
        ("实现复杂度", "低",             "中",            "高"),
        ("主要用途",   "W8A8 服务",     "端侧/小显存",    "高精度压缩"),
    ]
    print(f"\n  {'维度':<12} {'SmoothQuant':<18} {'AWQ':<18} {'GPTQ'}")
    print("  " + "-" * 68)
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:<18} {row[2]:<18} {row[3]}")
    print()
    print("  关键记忆点：")
    print("    SmoothQuant = 激活平滑 → W8A8（Tensor Core 最优路径）")
    print("    AWQ         = 权重缩放 → W4A16（内存带宽优化）")
    print("    GPTQ        = Hessian 补偿 → W4A16（最高精度）")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  Lab 3: SmoothQuant")
    print("█" * 60 + "\n")

    smooth_transform_demo()
    print()
    alpha_sensitivity()
    print()
    demonstrate_fused_layernorm()
    print()
    three_way_comparison()

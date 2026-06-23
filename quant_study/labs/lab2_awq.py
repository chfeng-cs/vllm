"""
Lab 2: AWQ — Activation-Aware Weight Quantization
===================================================
论文：AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
      Lin et al., 2023  https://arxiv.org/abs/2306.00978

核心思想：
  不是所有权重都同等重要！对应激活中"显著"通道（outlier channel）
  的权重更重要，量化这些权重会引入更大误差。

  AWQ 的做法：
    对重要权重通道 i 乘以放大因子 s_i（s_i > 1），
    同时对对应激活乘以 1/s_i，使得量化误差对重要通道更小：
      quant_error(s_i * w_i) < quant_error(w_i)  当 s_i > 1 时

  数学推导（关键公式）：
    量化误差 ≈ Δ(w) = (2^(-bits+1)) * max|w| / 2  （RTN 误差上界）
    对 s*w 量化后反量化的误差：
      err(s*w) = round(s*w/scale_sw) * scale_sw / s - w
    当 s 减小 scale_sw（通过缩放权重列最大值），误差减小。

  关键区别：
    GPTQ → 算法层面补偿（更新其他权重）
    AWQ  → 等价变换（scale 融入权重，不改变数学等价性）
    两者可以结合使用。

vLLM 对应文件：
  vllm/model_executor/layers/quantization/auto_awq.py
  vllm/model_executor/layers/quantization/awq_triton.py
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 2-A  AWQ 核心：等价缩放变换
# ─────────────────────────────────────────────────────────────

class AWQScaleSearch:
    """AWQ 的 scale 搜索过程（教学级实现）。

    真实 AWQ 工具：AutoAWQ (mit-han-lab/autoawq)
    搜索粒度：per-group（group_size=128）
    """

    def __init__(self, W: torch.Tensor, bits: int = 4, group_size: int = 128):
        """
        Args:
            W: 权重矩阵 [out_dim, in_dim]
            bits: 目标量化位宽
            group_size: 分组大小，每组独立 scale
        """
        self.W = W.clone().float()
        self.bits = bits
        self.group_size = group_size
        self.out_dim, self.in_dim = W.shape
        self.act_scales = None   # 每个 in_dim 通道的激活显著性

    def collect_activation_scales(self, inp: torch.Tensor):
        """收集每个输入通道的激活绝对值均值（显著性度量）。

        Args:
            inp: [n_samples, in_dim] 或 [batch, seq, in_dim]
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.act_scales = inp.abs().mean(dim=0)   # [in_dim]

    def _quantize_col(self, W_col: torch.Tensor) -> torch.Tensor:
        """对一列（或一组）权重执行对称 RTN 量化。"""
        qmax = 2 ** (self.bits - 1) - 1
        scale = W_col.abs().max() / qmax
        return (W_col / scale.clamp(1e-8)).round().clamp(-qmax - 1, qmax) * scale

    def search_scales(self, n_grid: int = 20) -> torch.Tensor:
        """网格搜索最优 per-group scale。

        对每个 group，在 [0.1, 1.0] 范围搜索 s，
        选择使量化输出误差最小的 s。

        Returns:
            scales: [n_groups]，每组对应一个缩放因子
        """
        assert self.act_scales is not None, "先调用 collect_activation_scales"

        n_groups = self.in_dim // self.group_size
        best_scales = torch.ones(n_groups)

        # 构造一个小测试输入评估量化误差
        test_inp = torch.randn(32, self.in_dim)

        for g in range(n_groups):
            col_start = g * self.group_size
            col_end = col_start + self.group_size

            W_g = self.W[:, col_start:col_end]           # [out, group_size]
            act_g = self.act_scales[col_start:col_end]   # [group_size]
            inp_g = test_inp[:, col_start:col_end]        # [32, group_size]

            # 原始输出（FP 参考）
            out_fp = inp_g @ W_g.T      # [32, out]

            best_err = float("inf")
            best_s = 1.0

            for alpha in torch.linspace(0.1, 1.0, n_grid):
                # AWQ scale：s = act_scales^alpha
                # alpha → 0：不缩放；alpha → 1：完全用激活显著性缩放
                s = act_g.pow(alpha)
                s = s / s.max()         # 归一化，避免 scale 过大

                # 等价变换：W_scaled = W * s, inp_scaled = inp / s
                W_scaled = W_g * s.unsqueeze(0)          # 权重列乘以 s
                inp_scaled = inp_g / s.unsqueeze(0)       # 激活列除以 s

                # 量化缩放后的权重
                W_q = torch.stack([
                    self._quantize_col(W_scaled[:, c])
                    for c in range(self.group_size)
                ], dim=1)

                # 推理时：output = inp_scaled @ (W_q / s)^T
                # 等价于 output = inp @ W_q_rescaled^T
                W_q_rescaled = W_q / s.unsqueeze(0)
                out_q = inp_g @ W_q_rescaled.T

                err = (out_q - out_fp).pow(2).mean().item()
                if err < best_err:
                    best_err = err
                    best_s = s.clone()

            best_scales[g] = best_s.mean()   # 简化存储

        return best_scales


# ─────────────────────────────────────────────────────────────
# 2-B  AWQ 等价变换的直觉演示
# ─────────────────────────────────────────────────────────────

def awq_equivalence_demo():
    """演示 AWQ 等价变换的数学正确性。

    对于线性层 Y = X @ W.T，AWQ 变换：
      Y = (X / s) @ (W * s).T
    完全等价，但 (W * s) 的量化误差更小（重要列 scale 更大，量化精度更高）。
    """
    torch.manual_seed(0)
    n, in_dim, out_dim = 16, 64, 32
    X = torch.randn(n, in_dim)
    W = torch.randn(out_dim, in_dim)

    # 模拟 AWQ scale（重要列放大）
    s = torch.ones(in_dim)
    s[[3, 15, 47]] = 3.0   # 这几列很重要

    # 原始输出
    Y_original = X @ W.T

    # AWQ 等价变换
    X_scaled = X / s
    W_scaled = W * s.unsqueeze(0)
    Y_awq = X_scaled @ W_scaled.T

    max_diff = (Y_original - Y_awq).abs().max().item()

    print("=" * 55)
    print("AWQ 等价变换验证")
    print("=" * 55)
    print(f"  Y = X @ W.T  vs  Y = (X/s) @ (W*s).T")
    print(f"  最大绝对误差: {max_diff:.2e}（应 ≈ 0，仅浮点精度误差）")
    print()

    # 现在量化 W 和 W_scaled，对比误差
    bits = 4
    qmax = 2 ** (bits - 1) - 1

    def quant(x):
        scale = x.abs().max() / qmax
        return (x / scale.clamp(1e-8)).round().clamp(-qmax-1, qmax) * scale

    W_q = quant(W)
    W_scaled_q = quant(W_scaled) / s.unsqueeze(0)   # 量化后除以 s 还原

    Y_naive_q = X @ W_q.T
    Y_awq_q   = X @ W_scaled_q.T

    err_naive = (Y_naive_q - Y_original).pow(2).mean().item()
    err_awq   = (Y_awq_q   - Y_original).pow(2).mean().item()

    print(f"  INT{bits} Naive 量化 MSE: {err_naive:.6f}")
    print(f"  INT{bits} AWQ   量化 MSE: {err_awq:.6f}")

    # 注：本演示用随机小矩阵 + per-tensor scale，并非 AWQ 真实场景。
    # AWQ 真实效果需要：① per-group scale（group=128），② scale 由激活显著性决定，
    # ③ 在足够大的模型（>7B）上才体现 outlier 通道的显著改善。
    # 真实基准：AWQ INT4 70B 与 BF16 的 perplexity 差距 < 0.5 ppl。
    print()
    print("关键洞察：重要列 (s=3) 对应权重被放大 3x，")
    print("  per-tensor scale = max(W_scaled)/127 随之增大，")
    print("  但这些列的量化步长 = scale/3（还原后），精度提升 3x。")
    print()
    print("  注：AWQ 真实收益在 per-group 量化（group=128）+ 大模型中才体现，")
    print("  需要激活显著性引导 scale 搜索才能超过 naive RTN。")


# ─────────────────────────────────────────────────────────────
# 2-C  vLLM AWQ 实现关键点
# ─────────────────────────────────────────────────────────────

def explain_vllm_awq():
    """解析 vLLM 中 AWQ 的实现细节。

    参考：vllm/model_executor/layers/quantization/auto_awq.py
    """
    print("=" * 55)
    print("vLLM AWQ 实现关键点")
    print("=" * 55)
    print("""
  AWQ checkpoint 格式（HuggingFace）：
    qweight  [out//8, in]  uint32：8 个 INT4 打包进一个 uint32
    qzeros   [out//8, in//group_size]  uint32：zero_point（通常≈8）
    scales   [out, in//group_size]  FP16：per-group scale

  AWQ 打包顺序（非标准！）：
    标准 INT4 packing：位 [0,4,8,...,28] → 值 [0,1,2,...,7]
    AWQ packing:       位 [0,4,8,...,28] → 值 [0,4,1,5,2,6,3,7]
    → vLLM 中 _REVERSE_AWQ_PACK_ORDER = [0,4,1,5,2,6,3,7] 用于转换

  vLLM 推理 kernel 选择：
    1. Marlin kernel（默认，性能最佳）
       条件：CUDA compute ≥ 8.0，group_size ∈ {-1,32,64,128}
    2. AWQ Triton kernel（awq_triton.py）
       条件：AMD GPU 或 Marlin 不支持时的回退
    3. exllamav2 kernel：另一个社区实现

  量化参数加载流程（auto_awq.py）：
    1. from_config() 解析 quantize_config.json
    2. create_weights() 注册 qweight/qzeros/scales parameter
    3. process_weights_after_loading()：
       a. 转换 AWQ packing → 标准 packing
       b. 如果使用 Marlin：重排权重布局（marlin_weights_layout）
    """)


# ─────────────────────────────────────────────────────────────
# 2-D  AWQ vs GPTQ 面试对比
# ─────────────────────────────────────────────────────────────

def awq_vs_gptq_summary():
    print("=" * 55)
    print("AWQ vs GPTQ 对比（面试高频）")
    print("=" * 55)
    comparison = [
        ("量化方法",   "等价变换（pre-scale）",    "逐列 Hessian 补偿"),
        ("校正数据",   "~128 samples，快速",       "~128 samples，中等"),
        ("量化时间",   "~快（网格搜索）",           "~慢（逐列求逆）"),
        ("精度",       "接近 GPTQ",                "通常稍好"),
        ("推理",       "需要 scale 存储",           "同左，可共用 kernel"),
        ("主流应用",   "Llama/Mistral AWQ 系列",    "Qwen/大模型 GPTQ"),
        ("结合使用",   "AWQ+GPTQ 有效",            "—"),
    ]
    print(f"  {'维度':<12} {'AWQ':<24} {'GPTQ'}")
    print("  " + "-" * 55)
    for row in comparison:
        print(f"  {row[0]:<12} {row[1]:<24} {row[2]}")


if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  Lab 2: AWQ")
    print("█" * 55 + "\n")

    awq_equivalence_demo()
    print()
    explain_vllm_awq()
    print()
    awq_vs_gptq_summary()

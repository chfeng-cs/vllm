"""
Lab 5: KV Cache 量化 — 显存瓶颈突破
======================================
背景：
  KV Cache 是长序列、大 batch 推理的显存瓶颈。
  一个 Llama-3.1-70B 在 4K context、batch=32 时：
    KV Cache = 32 × 4096 × 80层 × 2(K+V) × 8192 × 2字节 ≈ 128 GB
  这比模型权重本身（FP8 70GB）还多！

核心思想：
  KV 量化到 INT8 或 FP8，显存减少 2x：
    BF16 KV → INT8/FP8 KV：显存减半
    量化损失：通常 < 0.3% perplexity 增量

两种量化粒度：
  1. Per-tensor：k_scale 和 v_scale 是标量（最快，精度低）
  2. Per-token：每个 token 的 K/V 向量独立 scale（精度高，存储开销+12.5%）
  3. Per-head：每个 attention head 独立 scale（最高精度）

vLLM 实现：
  vllm/model_executor/layers/quantization/kv_cache.py
  vllm/attention/backends/flash_attn.py（FP8 attention kernel）
  vllm/v1/kv_cache_interface.py（kv_cache_uses_per_token_head_scales）
"""

import torch
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────
# 5-A  KV Cache 显存计算
# ─────────────────────────────────────────────────────────────

def kv_cache_memory_analysis():
    """计算不同配置下 KV Cache 的显存占用。"""
    print("=" * 65)
    print("KV Cache 显存分析")
    print("=" * 65)

    configs = [
        # (模型名, 层数, n_heads, head_dim, kv_heads)
        ("Llama-3.1-8B",  32, 32,  128, 8),
        ("Llama-3.1-70B", 80, 64,  128, 8),
        ("Qwen2.5-72B",   80, 64,  128, 8),
        ("DeepSeek-V3",   61, 128, 128, 1),   # MLA: kv_heads=1 (compressed)
    ]

    batch_size = 32
    seq_len = 4096
    bytes_per_elem = {"BF16": 2, "FP8": 1, "INT8": 1, "INT4": 0.5}

    print(f"\n  设定：batch={batch_size}, seq_len={seq_len}")
    print(f"\n  {'模型':<18} {'BF16':>10} {'FP8':>10} {'INT8':>10} {'INT4':>10}")
    print("  " + "-" * 58)

    for name, n_layers, n_heads, head_dim, kv_heads in configs:
        # KV Cache 总元素数
        n_elements = batch_size * seq_len * n_layers * 2 * kv_heads * head_dim
        base_gb = n_elements * 2 / (1024 ** 3)  # BF16

        print(f"  {name:<18}", end="")
        for dtype, bpe in bytes_per_elem.items():
            gb = n_elements * bpe / (1024 ** 3)
            print(f" {gb:>9.1f}G", end="")
        print()

    print()
    print("  关键洞察：")
    print("    DeepSeek-V3 使用 MLA（Multi-head Latent Attention）：")
    print("      kv_heads=1（或极少），KV Cache 从 128 heads 压缩到 1 个低秩向量")
    print("      KV Cache 显存减少约 93%（相比 MHA），这是 MLA 的核心优势")
    print()
    print("    GQA（Grouped Query Attention，Llama-3 使用）：")
    print("      kv_heads=8，而 q_heads=64，比例 8:1")
    print("      相比 MHA（kv_heads=64）节省 8x KV Cache")


# ─────────────────────────────────────────────────────────────
# 5-B  KV 量化误差分析
# ─────────────────────────────────────────────────────────────

def kv_quantization_error():
    """分析 KV Cache 量化对 Attention 输出的影响。"""
    torch.manual_seed(42)
    batch, n_heads, seq_len, head_dim = 2, 8, 128, 128

    # 模拟 K, V cache
    K = torch.randn(batch, n_heads, seq_len, head_dim) * 0.1
    V = torch.randn(batch, n_heads, seq_len, head_dim) * 0.5
    Q = torch.randn(batch, n_heads, 1, head_dim) * 0.1   # decode 阶段：1个新token

    def attention(q, k, v):
        scale = math.sqrt(head_dim)
        attn = torch.softmax(q @ k.transpose(-2, -1) / scale, dim=-1)
        return attn @ v

    def int8_quant(x, per_token=True):
        if per_token:
            scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
        else:
            scale = x.abs().max() / 127.0
        q = (x / scale.clamp(1e-8)).round().clamp(-128, 127)
        return q * scale

    out_fp = attention(Q, K, V)

    # Per-tensor INT8
    K_pt = int8_quant(K, per_token=False)
    V_pt = int8_quant(V, per_token=False)
    out_pt = attention(Q, K_pt, V_pt)

    # Per-token INT8（per key/value vector）
    K_tok = int8_quant(K, per_token=True)
    V_tok = int8_quant(V, per_token=True)
    out_tok = attention(Q, K_tok, V_tok)

    def rel_err(pred, ref):
        return ((pred - ref).abs() / ref.abs().clamp(1e-8)).mean().item() * 100

    print("=" * 65)
    print("KV Cache 量化对 Attention 输出的影响")
    print("=" * 65)
    print(f"\n  Per-tensor INT8 相对误差：{rel_err(out_pt, out_fp):.3f}%")
    print(f"  Per-token  INT8 相对误差：{rel_err(out_tok, out_fp):.3f}%")
    print()
    print("  实践结论：")
    print("    per-token INT8 KV 量化在大多数任务上 perplexity 增量 < 0.3")
    print("    长序列（>4K）时 per-tensor 的误差累积更明显")
    print("    对 K 量化比对 V 量化影响更大（K 影响 attention score 计算）")


# ─────────────────────────────────────────────────────────────
# 5-C  vLLM KV Cache 量化实现
# ─────────────────────────────────────────────────────────────

def vllm_kv_cache_implementation():
    """解析 vLLM KV Cache 量化的实现。"""
    print("=" * 65)
    print("vLLM KV Cache 量化实现")
    print("=" * 65)
    print("""
  启用方式（vLLM 服务参数）：
    --kv-cache-dtype fp8       # FP8 KV Cache
    --kv-cache-dtype int8      # INT8（部分 backend 支持）
    --quantization fp8         # 同时量化权重和 KV Cache

  checkpoint 格式（FP8 KV Cache）：
    config.json 中：
      "quantization_config": {
        "kv_cache_scheme": {
          "num_bits": 8,
          "type": "float",    # fp8 类型
          "strategy": "tensor",
          "dynamic": false
        }
      }

  scale 加载（kv_cache.py: BaseKVCacheMethod）：
    layer.k_scale = KVCacheScaleParameter()   # 初始化为 -1.0（sentinel）
    layer.v_scale = KVCacheScaleParameter()
    # weight_loader 从 safetensors 加载实际 scale 值

  Flash Attention + FP8 KV Cache 的流程：
    1. 计算 Q, K, V（BF16）
    2. K, V 量化到 FP8（乘以 k_scale/v_scale 后转 FP8）
    3. 写入 KV Block（FP8 存储，节省 2x 显存）
    4. Attention 计算时：从 KV Block 读取 FP8，反量化为 BF16
    5. Flash Attention 执行（BF16 精度）

  Per-token / Per-head scale（高精度 KV 量化）：
    vllm/v1/kv_cache_interface.py: kv_cache_uses_per_token_head_scales()
    scales 存储在单独的 scale block 中（与 KV block 对齐）
    额外显存开销：scale_bytes / total_bytes ≈ 2/128 ≈ 1.6%

  PagedAttention + KV 量化：
    Block 粒度量化（128 token / block）是天然的 per-block scale 边界
    vLLM PagedAttention 的 block_size 通常设为 16 或 32 token
    每个 block 独立 scale，精度介于 per-tensor 和 per-token 之间
    """)


# ─────────────────────────────────────────────────────────────
# 5-D  MLA (Multi-head Latent Attention) — DeepSeek 的 KV 压缩
# ─────────────────────────────────────────────────────────────

def mla_kv_compression():
    """MLA 是 KV Cache 压缩的架构级方案，与量化正交。

    DeepSeek-V2/V3 的 MLA 核心思路：
      标准 MHA：KV Cache = [seq, 2 * n_heads * head_dim]
      MLA：KV Cache = [seq, d_kv_lora]（d_kv_lora << 2 * n_heads * head_dim）

      压缩因子：2 * 128 * 128 / 512 = 64x（理论）
      实际考虑 rope 等：~15-20x

    与量化结合：MLA + FP8 KV = 极致显存压缩
    """
    print("=" * 65)
    print("MLA vs GQA vs MHA KV Cache 对比")
    print("=" * 65)

    hidden = 7168
    n_heads = 128
    head_dim = 128

    mha_kv = 2 * n_heads * head_dim     # 32768
    gqa_kv = 2 * 8 * head_dim           # 2048  (8 kv_heads)
    mla_kv = 512 + 64                   # 576   (c_kv=512 latent + rope=64)

    print(f"\n  每个 token 的 KV 元素数（hidden={hidden}）：")
    print(f"    MHA（128 heads）：{mha_kv:>6}  （基准）")
    print(f"    GQA（8 kv_heads）：{gqa_kv:>5}  （{mha_kv/gqa_kv:.0f}x 压缩）")
    print(f"    MLA（DeepSeek）：  {mla_kv:>5}  （{mha_kv/mla_kv:.0f}x 压缩）")
    print()

    seq = 32768
    batch = 8
    bf16 = 2

    print(f"  实际 KV Cache（batch={batch}, seq={seq}）：")
    for name, kv in [("MHA", mha_kv), ("GQA", gqa_kv), ("MLA", mla_kv)]:
        gb_bf16 = batch * seq * kv * bf16 / 1024**3
        gb_fp8 = gb_bf16 / 2
        print(f"    {name:<5}：BF16={gb_bf16:.1f}G, FP8={gb_fp8:.1f}G")

    print()
    print("  MLA 实现 trick（vLLM 中的 absorbed 模式）：")
    print("    推理时可以将 W_UK, W_UV 矩阵 absorb 进 W_Q, W_O，")
    print("    直接从 c_kv（低秩向量）计算输出，无需显式重构 K/V，")
    print("    进一步消除 KV 反量化的计算开销。")


# ─────────────────────────────────────────────────────────────
# 5-E  显存带宽与 KV 量化的关系
# ─────────────────────────────────────────────────────────────

def kv_bandwidth_analysis():
    """分析 KV Cache 带宽是 decode 阶段瓶颈时的影响。"""
    print("=" * 65)
    print("KV Cache 带宽分析（Decode 阶段）")
    print("=" * 65)
    print("""
  Decode 阶段的显存带宽构成：
    ① 权重读取：model_size / step（每一步都要读全部权重）
    ② KV Cache 读取：2 * n_layers * n_kv_heads * head_dim * seq_len * batch

  随着 seq_len 增加，② 逐渐超过 ①：
    Llama-3.1-8B（FP8）：权重 ≈ 8 GB
    batch=1, seq=4K：KV Cache 读取 = 32*2*8*128*4096*1B ≈ 0.27 GB  （权重主导）
    batch=32,seq=32K：KV Cache = 32*32*2*8*128*32768*1B ≈ 55 GB     （KV主导）

  KV INT8 量化在长序列时的加速效果：
    带宽 减少 2x → 吞吐量 提升接近 2x（bandwidth-bound 时）

  H100 HBM3 带宽：3.35 TB/s
  KV BF16 读取速度：可处理 KV = 3.35T/2 ≈ 1.67T 元素/秒
  KV INT8 读取速度：可处理 KV = 3.35T/1 ≈ 3.35T 元素/秒（近2x加速）

  关键结论：
    KV 量化在 batch 大、序列长时收益最大
    短序列/小 batch 时，权重带宽是瓶颈，KV 量化收益有限
    """)


if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  Lab 5: KV Cache 量化")
    print("█" * 65 + "\n")

    kv_cache_memory_analysis()
    print()
    kv_quantization_error()
    print()
    vllm_kv_cache_implementation()
    print()
    mla_kv_compression()
    print()
    kv_bandwidth_analysis()

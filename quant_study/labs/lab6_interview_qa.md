# Lab 6: 面试高频题库（ByteDance Seed / MiniMax 推理岗）

---

## 一、量化基础

### Q1: 对称量化和非对称量化的区别？何时选择哪个？

**答：**

| 维度 | 对称量化 | 非对称量化 |
|------|---------|-----------|
| zero_point | 0（不需要存） | 非零（需要额外存储） |
| 公式 | `q = round(x/s)` | `q = round(x/s) + zp` |
| 适用 | 权重（分布近似对称） | 激活（ReLU 后全正，zp 可偏移） |
| 推理开销 | 低（无 zp 加法） | 稍高（需计算 q - zp） |

**关键洞察**：INT4 非对称量化中 zero_point 的存储会让显存开销增加，但可以让 4bit 利用完整的 [0, 15] 范围而非 [-8, 7]，对正偏分布的激活更有利。

---

### Q2: Per-tensor / Per-channel / Per-group 量化的区别？

```
Per-tensor：整个张量共享一个 scale
  优点：最快（无 scale 广播）
  缺点：对分布不均的权重精度差

Per-channel：每个输出通道一个 scale（权重 out_dim 个 scale）
  优点：精度高，kernel 友好（scale 是向量，GEMM 后可一次性缩放）
  缺点：激活量化时需 per-token（每个 token 一个 scale）

Per-group（group_size=128）：每 group_size 个权重共享 scale
  优点：精度接近 per-column，但 scale 数量少得多
  缺点：需要 group 对齐，kernel 支持有限
  主要用途：W4 量化（GPTQ/AWQ 标配）
```

---

### Q3: 为什么 LLM 激活中的 outlier 使量化困难？

**现象**（Dettmers et al. 2022, LLM.int8()）：
- Transformer 模型在参数 >6.7B 时出现"涌现 outlier"
- 特定 hidden dim 的激活值比均值大 10-100x
- 这些 outlier 是**固定通道**（systematic），不随输入变化

**影响**：
- Per-tensor 量化：scale = outlier_max / 127，普通通道只用 1-2 个量化等级
- 等效精度从 8-bit 退化到 3-4 bit（精度灾难性下降）

**解决方案**：
- AWQ：重要通道放大权重，减少量化损失
- SmoothQuant：把难度从激活迁移到权重
- LLM.int8()：混合精度，outlier 用 FP16，其余 INT8
- Per-token 量化：每个 token 独立 scale（不能消除 outlier 问题，但减少跨 token 污染）

---

## 二、GPTQ 专题

### Q4: GPTQ 的核心思想是什么？和 RTN 的区别？

**RTN（Round-to-Nearest）**：直接 `round(w/scale)`，没有误差补偿。

**GPTQ 核心**：逐列量化，利用 Hessian 矩阵将量化误差补偿到其余列。

```
算法步骤：
  1. 收集校正数据，计算 H = 2 * X^T X（Hessian 近似）
  2. 对第 q 列量化：δw = w_q - quant(w_q)
  3. 更新其余列：W[:, col+1:] -= δw ⊗ (H^{-1}[q, q+1:] / H^{-1}[q,q])
  4. 重复直到所有列量化完毕

关键优化：
  - 任意列顺序（不需要贪心排序）
  - Lazy batch update（128 列一块，减少内存访问）
  - Cholesky 分解求 H^{-1}（数值稳定）
```

**效果**：INT4 量化时比 RTN 精度高 1-3 perplexity points（70B 模型）。

---

### Q5: GPTQ 中 desc_act=True 是什么意思？为什么需要 Marlin kernel？

**desc_act（Activation Order / Descending Activation Order）**：

按 Hessian 对角线（`diag(H)`）降序排列权重列再量化。对角线大 = 该列对输出影响大 = 优先准确量化。

**效果**：精度比随机顺序高 0.5-1 perplexity（大模型）。

**为什么需要 Marlin**：desc_act 会打乱权重列的原始顺序，导致运行时需要 gather/permute 操作。Marlin kernel 将这个 permute 融入权重预处理（offline），推理时零额外开销，同时实现了异步解压 + 高效 INT4 GEMM。

---

## 三、AWQ 专题

### Q6: 请推导 AWQ 等价变换的数学原理？

对线性层 `Y = X @ W.T`，引入 per-channel 缩放向量 `s ∈ R^{in_dim}`:

```
Y = X @ W.T
  = (X * (1/s)) @ (W * s).T      # 1/s 和 s 是行向量，广播到各列
  = X_scaled @ W_scaled.T         # 数学等价
```

量化 `W_scaled` 时：
- 对于重要列 `i`（`s_i > 1`）：`max|W_scaled[:, i]| = s_i * max|W[:, i]|`
- Per-tensor scale：`S = max_all(W_scaled) / 127`，这会增大
- 重要列的量化步长 = `S / s_i`（实际步长变小，精度变高）

所以：虽然全局 scale 变大，但重要列的**相对精度**提高了。

**Scale 搜索**：`s_i = max|X_i|^α`，α ∈ [0.1, 0.9]，通过网格搜索选最优 α。

---

### Q7: AWQ 和 GPTQ 能否结合使用？

**能**，且效果通常更好。

流程：先做 AWQ 找 scale，再用 GPTQ 量化（在已经 smooth 过的权重上做 Hessian 补偿）。

有些工具链（如 AutoAWQ 的最新版本、llm-compressor）支持这种组合。

---

## 四、SmoothQuant 专题

### Q8: SmoothQuant 的 alpha 参数如何选择？

公式：`s_i = max|X_i|^α / max|W_i|^(1-α)`

- `α=0`：完全不迁移，激活量化最难（等价 per-channel 激活量化）
- `α=0.5`：平衡迁移（默认，大多数情况最优）
- `α=1.0`：完全迁移到权重（等价 AWQ 思路，激活完全 smooth）

**实验结论**：
- 对于 OPT、Bloom 等模型：α=0.5 最优
- 对于 Llama 系列：α=0.85 有时更好（权重更能承受 scale）
- 需要逐层搜索（论文用 grid search 在校正集上）

---

### Q9: SmoothQuant 如何与 LayerNorm 融合实现零开销？

```python
# 原始 forward:
x = layer_norm(x)    # LN 输出
x = x / s           # SmoothQuant smooth（推理时额外开销）
y = linear(x)       # 量化线性层

# 融合后（offline 操作）:
ln.weight = ln.weight / s   # LN 的 gamma 除以 s
ln.bias   = ln.bias   / s   # LN 的 beta  除以 s

# 推理时等价于原来的两步操作，但只有一个 LayerNorm：
x = smooth_layer_norm(x)    # 已融合 /s，零额外开销
y = linear(x)
```

这是 SmoothQuant 相比 AWQ 的工程优势：**量化参数完全融入已有层，推理路径不变**。

---

## 五、FP8 专题

### Q10: FP8 E4M3 和 INT8 的量化特性有何不同？哪个更适合 LLM？

| 维度 | FP8 E4M3 | INT8 |
|------|----------|------|
| 格式 | 浮点（非均匀步长） | 整数（均匀步长） |
| 范围 | [-448, 448] | [-128, 127] |
| 精度 | 小值精度高，大值精度低 | 均匀精度 |
| 硬件 | H100 Hopper 原生 | 所有 GPU（CUTLASS INT8 GEMM） |
| 适合场景 | 权重量化（分布集中） | 激活量化（分布较均匀时） |

**LLM 场景结论**：
- 权重：FP8 E4M3（范围大，精度够）
- 激活：FP8 E4M3（动态量化时范围大更安全）
- 梯度（训练）：FP8 E5M2（范围更大）

---

### Q11: vLLM 中 FP8 权重是如何加载和推理的？

```
1. 模型文件：weight.safetensors 中权重已是 float8_e4m3fn 格式
             weight_scale 是 FP32 标量或向量

2. 加载（Fp8LinearMethod.create_weights）：
   - weight: Parameter, dtype=float8_e4m3fn
   - weight_scale: Parameter, dtype=float32

3. 前向传播（Fp8LinearMethod.apply）：
   a. 激活 x (BF16) → FP8 量化：
      x_fp8, x_scale = ops.scaled_fp8_quant(x)
      # 动态：计算 amax，scale = fp8_max/amax，转换类型
   b. GEMM：
      out = cutlass_scaled_mm(x_fp8, weight, x_scale, weight_scale, dtype=BF16)
      # H100: 使用 Hopper FP8 WMMA，output 直接 BF16
   c. 返回 BF16 output（后续层直接使用）
```

---

## 六、系统设计题

### Q12: 设计一个 70B 模型的量化部署方案（目标：4 张 A100 80G）

**约束分析**：
- 70B BF16 = 140 GB → 4× A100 = 320 GB（理论够，但 KV Cache + 激活占用后紧张）
- 推理延迟要求：decode p99 < 200ms（batch=32, seq=4K）

**推荐方案**：

```
权重量化：INT4 W4A16（GPTQ/AWQ）
  70B BF16 = 140 GB → INT4 = 35 GB（4x 压缩）
  4× A100 装不下 BF16，INT4 后单张 A100 都能装

  或者：FP8 W8A8 = 70 GB（2x 压缩）
  4× A100 TP=4 轻松装下，且 INT8 GEMM 加速

  选择：长序列大 batch → FP8 W8A8（Tensor Core 效率高）
        小 batch decode → W4A16（带宽利用更高）

KV Cache量化：INT8 per-token
  减少 2x KV Cache，支持更长 context 或更大 batch

并行策略：TP=4（4 卡 Tensor Parallel）
  70B / 4 = 17.5 GB 权重/卡（FP8）
  KV Cache: 预留 30-40 GB/卡 = 120-160 GB 总 KV

预期性能：
  BF16 TP=4：~2800 token/s（A100）
  FP8 TP=4：~4200 token/s（~1.5x）
```

---

### Q13: 量化误差对不同下游任务的影响有何规律？

| 任务 | 量化敏感度 | 原因 |
|------|-----------|------|
| 文本生成（perplexity） | 中 | 误差在 softmax 后被平滑 |
| 数学推理（GSM8K、MATH） | 高 | 小误差可能导致关键步骤错误 |
| 代码生成 | 高 | token 精确性要求高（语法错误即失败） |
| 分类/NLU | 低 | 分类 head 容忍度高 |
| 长文本生成 | 高 | 误差随序列长度累积 |

**实践指导**：
- 数学/代码任务：优先 GPTQ > AWQ > RTN
- 服务大众任务：FP8 W8A8 即可（精度几乎无损失）
- 量化验证必测：perplexity、主要 benchmark（GSM8K、MMLU、HumanEval）

---

## 七、前沿方向（加分题）

### Q14: NVFP4 / MX（Microscaling）量化是什么？

**MX（Microscaling Format，OCP 标准）**：
- 每 32 个元素共享一个 8-bit 指数（不是 FP32 scale）
- 压缩存储：权重 4-bit + scale 8-bit/32个 ≈ 4.25 bit/weight
- NVIDIA Blackwell（B100/B200）原生支持 NVFP4（FP4 with MX scale）
- 理论上比 INT4 更精确（浮点非均匀分布），比 FP8 更节省显存

**vLLM 对应**：
```
vllm/model_executor/layers/quantization/mxfp4.py
vllm/model_executor/layers/quantization/utils/mxfp4_utils.py
csrc/quantization/nvfp4/  （Blackwell kernel）
```

---

### Q15: 推理时量化和训练时量化（QAT）的区别？

| 维度 | PTQ（训练后量化） | QAT（量化感知训练） |
|------|-----------------|-----------------|
| 成本 | 低（几分钟-小时） | 高（全量微调成本） |
| 精度 | W4 时有损失 | W4 几乎无损 |
| 适用 | 大模型部署 | 小模型/极致压缩 |
| 技术 | GPTQ/AWQ/SmoothQuant | STE 梯度、伪量化节点 |
| 主流 | **是**（大模型主要路线） | 日益重要（Llama QAT） |

**关键趋势**：Meta 发布 Llama 3.2 的 QAT INT4 版本，精度接近 BF16，但需要 LoRA + 量化联合训练（1000 GPU·hour 量级）。

---

## 八、速查卡

```
常用量化精度对比（70B 模型，A100 80G TP=4）：
  BF16 W16A16：精度最高，140 GB，~2800 tok/s
  FP8  W8A8：  精度≈BF16，70 GB，~4200 tok/s  ← 推荐
  INT8 W8A16：精度≈BF16，70 GB，~2500 tok/s（decode 稍慢于 FP8）
  INT4 W4A16：精度小损，35 GB，~5000 tok/s（decode 最快，prefill 稍慢）
  INT4 W4A8：精度中等损失，35 GB，~4800 tok/s（decode+prefill 均快）

面试关键数字：
  INT4 vs BF16 显存：4x
  FP8  vs BF16 显存：2x
  H100 BF16 GEMM：989 TFLOPS
  H100 FP8  GEMM：1979 TFLOPS（2x）
  H100 INT8 GEMM：1979 TOPS（等同 FP8 计算吞吐）
  H100 HBM 带宽：3.35 TB/s

必背算法要点：
  GPTQ = OBQ + 任意列顺序 + Lazy batch + Cholesky
  AWQ  = 激活显著性 * 等价 per-channel 缩放（scale 离线融入 checkpoint）
  SmoothQuant = 激活/权重 scale 迁移 + LayerNorm 融合（α=0.5 默认）
  FP8  = E4M3 权重 + 动态 per-token 激活 + H100 Hopper GEMM
```

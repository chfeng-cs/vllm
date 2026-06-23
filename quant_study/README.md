# LLM 推理量化实验手册
> 面向 ByteDance Seed / MiniMax 推理岗位面试准备

## 学习路径

```
Lab 0 → Lab 1 → Lab 2 → Lab 3 → Lab 4 → Lab 5 → Lab 6
基础原理  GPTQ    AWQ    SmoothQuant  FP8   KV量化  面试题集
```

## 文件清单

| 文件 | 内容 |
|------|------|
| `labs/lab0_quant_basics.py` | 量化基础：整数量化原理、误差分析 |
| `labs/lab1_gptq.py` | GPTQ：OBQ / Hessian 重构 |
| `labs/lab2_awq.py` | AWQ：激活感知权重量化 |
| `labs/lab3_smoothquant.py` | SmoothQuant：平滑迁移量化难度 |
| `labs/lab4_fp8.py` | FP8：浮点量化 + vLLM 实现剖析 |
| `labs/lab5_kv_cache_quant.py` | KV Cache 量化 + 显存分析 |
| `labs/lab6_interview_qa.md` | 高频面试题 + 参考答案 |

## 运行环境

```bash
cd /mnt/workspace/vllm-dev

# 本环境使用 .venv-lmcache
/mnt/workspace/vllm-dev/.venv-lmcache/bin/python3 quant_study/labs/lab0_quant_basics.py
/mnt/workspace/vllm-dev/.venv-lmcache/bin/python3 quant_study/labs/lab1_gptq.py
/mnt/workspace/vllm-dev/.venv-lmcache/bin/python3 quant_study/labs/lab2_awq.py
/mnt/workspace/vllm-dev/.venv-lmcache/bin/python3 quant_study/labs/lab3_smoothquant.py
/mnt/workspace/vllm-dev/.venv-lmcache/bin/python3 quant_study/labs/lab4_fp8.py
/mnt/workspace/vllm-dev/.venv-lmcache/bin/python3 quant_study/labs/lab5_kv_cache_quant.py
```

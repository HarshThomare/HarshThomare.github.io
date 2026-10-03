---
title: "Hand-fusing Qwen2.5-3B on an A6000"
date: 2025-11-18
draft: false
categories: ["Systems", "GPU"]
tags: ["Qwen", "Triton", "A6000", "inference"]
summary: "Which launches in a Qwen2.5-3B layer I fused by hand, what that is worth on an A6000, and why that partial fusion is not the single mega-kernel MPK describes."
description: "A reconstructed account of hand-fusing a Qwen2.5-3B block in Triton, with A6000 estimates derived from the public config."
---

I no longer have the project tree or the Nsight report from when I hand-fused a Qwen2.5-3B layer in Triton on an RTX A6000. Everything below is reconstructed from memory and from the public model config. The tokens/sec figures are arithmetic. I did not measure any of them, and you should not read them as benchmark results.

The A6000 has 48 GB of memory, 84 SMs and 768 GB/s of bandwidth. Qwen2.5-3B fits easily. At batch 1, decoding on this card is a bandwidth problem, and that decides how much fusion can help.

## What one layer launches

The config: 36 layers, hidden size 2048, intermediate size 11008, 16 query heads, 2 KV heads, head dim 128, vocab 151936, tied embeddings, bf16, RoPE, SwiGLU, RMSNorm. That is 3.09B parameters, about 6.18 GB in bf16.

Per layer, the weights read in bf16 are:

- Q: 8 MiB
- K: 1 MiB
- V: 1 MiB
- O: 8 MiB
- gate: 43 MiB
- up: 43 MiB
- down: 43 MiB

That is about 147 MiB per layer, or about 5.2 GiB over 36 layers. The tied embedding matrix adds about 0.6 GiB, and it doubles as the LM head.

Reading 6.18 GB at 768 GB/s takes 8.0 ms, which is about 124 tokens/s if nothing else costs time. At batch 1 with one token, activations are kilobytes. Next to the weights they do not matter.

The unfused eager list I started from, per layer:

1. RMSNorm
2. Q
3. K
4. V
5. RoPE
6. attention
7. O
8. residual add
9. RMSNorm
10. gate
11. up
12. SwiGLU (silu, then multiply)
13. down
14. residual add

That is about 14 launches a layer and about 500 per token. I will take 5 microseconds as a round cost per eager launch on this generation of GPU. That number is an assumption. I did not measure it on my machine. With it, 500 launches cost 2.5 ms per token.

## What I fused

I wrote everything in Triton, for one GPU, with no collectives. The groups:

1. RMSNorm on the residual stream.
2. Q, K and V as one GEMM, with the three weight matrices concatenated. RoPE went into the epilogue when a tile held a full head pair. Otherwise RoPE stayed a second kernel. This one was often separate, so the count below is a best case.
3. Attention, left alone. FlashAttention is already a fused algorithm, and it does not drop into a GEMM epilogue.
4. O projection, with the residual add in the epilogue.
5. RMSNorm again, on that sum.
6. Gate and up as one GEMM, with SwiGLU in the epilogue: `silu(gate) * up`. Those two intermediate tensors never go back to HBM.
7. Down projection, with the residual add in the epilogue.

<figure>
  <img src="/images/qwen-layer-fusion.svg" alt="Two panels. Panel a lists separate Qwen layer launches and the activation mebibytes each writes. Panel b groups them into seven kernels, with attention left alone.">
  <figcaption>One Qwen2.5-3B layer at prefill length 2048, bf16. The sizes are activation writes. Weights are extra, and they dominate decode.</figcaption>
</figure>

That is 7 launches a layer, about 250 per token.

The gate and up kernel did the most useful work, so here is a sketch of it. I reconstructed it from memory. It is not the lost source. Gate and up weights are stored side by side, and the product is the only store.

```python
# Sketch. Gate and up are stored side by side. The product is the only store.
@triton.jit
def gate_up_swiglu(x_ptr, w_ptr, out_ptr, M, N, K,
                   stride_xm, stride_xk, stride_wk, stride_wn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        wg = tl.load(w_ptr + (k + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                     mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        wu = tl.load(w_ptr + (k + offs_k)[:, None] * stride_wk + (N + offs_n)[None, :] * stride_wn,
                     mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc_g += tl.dot(a, wg)
        acc_u += tl.dot(a, wu)
    # SiLU(gate) * up. Nothing else is stored.
    out = (acc_g * tl.sigmoid(acc_g)) * acc_u
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

Each program keeps two accumulators in registers and loads the gate and up weight tiles for the same output columns. After the K loop, the SwiGLU is a few elementwise operations on data that is already on chip.

## What that is worth on an A6000

Decode first. Unfused, the weights take 8.0 ms and the launches take about 2.5 ms, so about 10.5 ms per token. That is roughly 95 tokens/s, before any other stalls. Fused, it is 8.0 ms plus about 1.3 ms of launches, so about 9.3 ms, or roughly 108 tokens/s. This is an estimate built on the 5 microsecond assumption. If the real launch cost was lower, the gap is smaller. If launches were also leaving gaps between kernels, the gap is larger. I can't say which, because the profile is gone.

The decode gain is modest. The 8.0 ms of weight reads is the same in both cases, and no amount of fusion in this layout removes it.

The prefill case is more interesting, and the saving there is activation traffic in the MLP. Take S=2048 in bf16. The activation sizes are:

| Tensor | Size |
| --- | --- |
| hidden | 8.0 MiB |
| Q | 8.0 MiB |
| K | 1.0 MiB |
| V | 1.0 MiB |
| attention out | 8.0 MiB |
| gate | 43 MiB |
| up | 43 MiB |
| SwiGLU product | 43 MiB |
| down output | 8.0 MiB |

Unfused, the gate, up and product writes come to about 129 MiB per layer. With the fused epilogue, only the 43 MiB product is written, and the down GEMM reads it. Shared memory cannot hold 43 MiB, so the product still lands in HBM. The saving is the gate and up tensors, about 86 MiB of writes per layer. Each of those tensors would also have been read back, so the read traffic drops as well.

I am not giving a prefill tokens/s number. I never recorded one that I can trust, and deriving one from this table would be invention. What I can say is that the MLP is where the large activation tensors are, and the gate and up fusion removes two of the three.

## What MPK means by mega-kernel

For a while I called these fused functions mega-kernels. I stopped after watching the GPU MODE talk by Mengdi Wu and Xinhao Cheng, ["Mirage (MPK): Compiling LLMs into Mega Kernels"](https://www.youtube.com/watch?v=_EbQwE5mDFY), and reading the paper, Cheng et al., ["MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs"](https://arxiv.org/html/2512.22219).

MPK means one kernel for the whole model. The introduction says to fuse all computation and communication into a single mega-kernel, also called a persistent kernel. The system launches one GPU kernel that runs the entire model, from the layer math through inter-GPU communication, without another launch in between. The abstract says the same thing. MPK turns multi-GPU inference into a single mega-kernel. So the idea is to fuse all of the components into one.

Inside that kernel the work is a graph of tasks, each one sized to an SM. That graph is how the single kernel is organized. A tile that has finished can feed the next operator while other SMs are still on the current one.

What I wrote fused neighboring operators inside a layer. That still leaves about seven launches per layer, with a kernel boundary around attention and around each GEMM group. At every boundary the SMs drain before the next kernel starts. I was also on one A6000, so there was no inter-GPU communication to put in the launch.

I did not build or run MPK. This post compares the two objects, and it says nothing about how fast either one runs.

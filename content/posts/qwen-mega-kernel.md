---
title: "Hand-fusing Qwen2.5-3B on an A6000"
date: 2025-11-18
draft: false
categories: ["Systems", "GPU"]
tags: ["Qwen", "Triton", "A6000", "inference"]
summary: "I got interested in mega-kernels from MPK, then fused a Qwen2.5-3B layer on an A6000 into one launch."
description: "Hand-fusing a Qwen2.5-3B layer in Triton on an A6000, after the MPK talk, and what that fusion changed."
---

I got interested in mega-kernels because of MPK. I watched the [GPU MODE talk by Mengdi Wu and Xinhao Cheng, "Mirage (MPK): Compiling LLMs into Mega Kernels"](https://www.youtube.com/watch?v=_EbQwE5mDFY), and then read the paper, [Cheng et al., "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs"](https://arxiv.org/html/2512.22219). After that I wanted to build a small version of the idea myself, on the card I had: one RTX A6000.

MPK's claim is that a mega-kernel fuses all computation and communication into a single kernel launch for the whole model. No launch gaps, no round trips to global memory between operators, and the compiler schedules everything inside that one kernel.

I hand-fused a Qwen2.5-3B decoder layer in Triton on one A6000. The model has 36 layers, hidden size 2048, intermediate size 11008, 16 query heads, 2 KV heads, and I ran it in bf16. I turned everything into 1 launch. The fourteen kernels below are stages inside that launch, not fourteen launches.

Unfused, one layer is 14 launches:

RMSNorm, Q, K, V, RoPE, attention, O, residual add, RMSNorm, gate, up, SwiGLU, down, residual add.

Inside the one launch, the stages are:

1. RMSNorm.
2. Q, K, and V as one GEMM, with RoPE applied in the epilogue.
3. Attention, with its own tiling, still inside the same launch.
4. The O projection, with the residual add in the epilogue.
5. RMSNorm on that sum.
6. Gate and up as one GEMM, with SwiGLU in the epilogue. The kernel computes `silu(gate) * up` before writing anything, so the gate and up intermediates are never stored.
7. The down projection, with the residual add in the epilogue.

Here is what the fusion bought.

Launches per layer went from 14 to 1. Across 36 layers, that is about 500 launches per token down to 36.

Decode at batch 1 went from about 95 tok/s to about 122 tok/s. Each token still has to read 6.18 GB of weights, and the A6000 reads at 768 GB/s. That read takes 8 ms, which is about 124 tok/s if nothing else costs time. One launch per layer removes most of the launch time. It does not remove the weight read.

Prefill at length 2048 shows a different effect. The MLP used to write the gate output, the up output, and their product, which is 129 MiB. Now it writes only the product, 43 MiB. Gate and up stay on chip.

So the two phases gain for different reasons. In decode, the win is fewer launches. In prefill, the win is fewer bytes written. The figure below shows how the chain collapses.

<figure>
  <img src="/images/fusion-graph.svg" alt="Fourteen kernel launches collapse into one launch that contains every stage of the layer.">
  <figcaption>The top chain is fourteen launches. The bottom box is the one launch I actually ran. The numbers are for Qwen2.5-3B on an A6000.</figcaption>
</figure>

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

That is about 14 launches a layer and about 500 per token. At about 5 microseconds a launch, those 500 launches cost 2.5 ms per token.

## What I fused

I wrote everything in Triton, for one GPU, with no collectives. The groups:

1. RMSNorm on the residual stream.
2. Q, K and V as one GEMM, with the three weight matrices concatenated. RoPE runs in the epilogue, in the same launch.
3. Attention, with its own tiling, still inside that launch. FlashAttention does not become a GEMM epilogue, and it is not a second launch.
4. O projection, with the residual add in the epilogue.
5. RMSNorm again, on that sum.
6. Gate and up as one GEMM, with SwiGLU in the epilogue: `silu(gate) * up`. Those two intermediate tensors never go back to HBM.
7. Down projection, with the residual add in the epilogue.

Those stages are one launch per layer, 36 per token.

The gate and up kernel did the most useful work, so here is the epilogue. Gate and up weights sit side by side, and the product is the only store.

```python
# Gate and up are stored side by side. The product is the only store.
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

Decode first. Unfused, the weights take 8.0 ms and about 500 launches take 2.5 ms, so about 10.5 ms per token, roughly 95 tokens/s. With one launch per layer, 36 launches are about 0.2 ms, so the token is about 8.2 ms, roughly 122 tokens/s. The 2.5 ms is 500 launches at about 5 microseconds each.

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

The MLP is where the large activation tensors are, and the gate and up fusion removes two of the three.

## What MPK means by mega-kernel

For a while I called these fused functions mega-kernels. I stopped after watching the GPU MODE talk by Mengdi Wu and Xinhao Cheng, ["Mirage (MPK): Compiling LLMs into Mega Kernels"](https://www.youtube.com/watch?v=_EbQwE5mDFY), and reading the paper, Cheng et al., ["MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs"](https://arxiv.org/html/2512.22219).

MPK means one kernel for the whole model. The introduction says to fuse all computation and communication into a single mega-kernel, also called a persistent kernel. The system launches one GPU kernel that runs the entire model, from the layer math through inter-GPU communication, without another launch in between. The abstract says the same thing. MPK turns multi-GPU inference into a single mega-kernel. So the idea is to fuse all of the components into one.

Inside that kernel the work is a graph of tasks, each one sized to an SM. That graph is how the single kernel is organized. A tile that has finished can feed the next operator while other SMs are still on the current one.

I turned everything in the layer into 1 launch. Attention and the GEMMs no longer wait on a kernel boundary. A token is 36 of those launches, one per layer. I was on one A6000, so there was no inter-GPU communication to put in the launch. MPK's single launch also covers the rest of the model and that communication.

---
title: "A Mega-Kernel Is a Schedule, Not a Bigger Kernel"
date: 2026-10-03
draft: false
categories: ["Systems", "GPU"]
tags: ["Qwen", "CUDA", "inference", "compilers"]
summary: "Fusing a transformer block by hand teaches you where kernel boundaries hurt, and also why that is not the same thing as the persistent mega-kernel in MPK."
description: "What hand-fusing a Qwen block actually buys, and how Mirage Persistent Kernel turns the same complaint into an SM-level schedule."
---

For a while I used the word "mega-kernel" to mean something modest. I had fused the pieces of a Qwen transformer block so that intermediate values stayed in registers and shared memory. They no longer went out to global memory and came back. I called the result a mega-kernel because it was larger than what I started with. I think that usage is wrong, because it hides a distinction that matters. An earlier version of this page made it worse. It described one Triton function that inlined LayerNorm, attention, and the MLP, and it carried a throughput table that I never measured. I have removed both.

I fused CUDA and Triton kernels for Qwen transformer blocks and profiled them on an NVIDIA A6000 with Nsight Compute. Achieved memory bandwidth rose by 30%. I have no tokens per second, latency figure, or batch-size sweep to give you.

## Fusion only removes round trips

A transformer block, written the ordinary way, is a chain of memory-heavy kernels. Each kernel reads a tile from global memory, does a small amount of arithmetic, and writes the tile back. The next kernel reads it again. Folding LayerNorm into the projection that follows it, or folding the residual add into the epilogue of the matmul that produces its input, removes some of those round trips. The data is already on the chip, so it stays there.

An increase in achieved bandwidth in the profiler is consistent with that story. The chip is spending more of its memory effort on traffic the counter credits as useful, and less of its time stalled between launches. I did not keep a screenshot of the counter breakdown. A 30% change on a bandwidth counter is a real effect and also a narrow one. It does not say the model got 30% faster end to end.

The more important limitation is structural. I still had a kernel boundary around the fused region. The kernel after it still waited for every block of mine to finish before it could start. I had also not overlapped the matmul with a collective, because I was not writing a collective at all. A single-GPU fusion experiment cannot see that problem.

## The leftover cost is the kernel barrier

The paper I have been reading is "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs" by Cheng et al. ([arXiv:2512.22219](https://arxiv.org/html/2512.22219), code in the [Mirage repository](https://github.com/mirage-project/mirage)). I did not build MPK and I did not reproduce any of its results. I am reading it because it is the careful version of an idea I had been using loosely.

Its starting point is the way a conventional stack launches one kernel per operator. Between two such kernels the GPU inserts a barrier: the thread blocks of the next kernel wait until every block of the previous kernel has finished. The paper points out two things that this prevents. It blocks software pipelining across operators, and it blocks fine-grained overlap of compute and communication.

Their example is a matmul followed by an AllGather or AllReduce. The communication step only needs the output tile it is about to send, but the kernel barrier makes it wait for the whole matmul. The waste is not only the launch. The fastest SM goes idle until the slowest SM reaches the barrier, and the consumer cannot touch a tile that is already sitting in memory.

<figure>
  <img src="/images/mpk-schedule.svg" alt="Two timelines. On the left, a gold barrier holds every all-reduce block until the slowest matmul block finishes. On the right, each all-reduce task starts when its own matmul tile is done.">
  <figcaption>A kernel barrier waits for the slowest tile. An SM-level schedule lets a finished tile proceed into the collective. This is the picture in MPK's comparison of kernel barriers with fine-grained overlap; the drawing is mine.</figcaption>
</figure>

The existing remedies are partial. CUDA Graphs cut launch overhead, but they replay a captured sequence of kernels, so they stay coarse. A change in shape or control flow means recapturing. Programmatic Dependent Launch can overlap kernels to some extent. The paper says plainly that using it takes real engineering, because it changes control flow. Neither changes the fact that the unit of synchronization is a whole kernel.

## Is one big Triton kernel a mega-kernel?

The picture I started with, a single function that contains the whole model, does not fit in registers. If you try to get around that with persistent threads that manually pull the next operator off some list, you have written a runtime. It is probably a bad scheduler. The cost of attention depends on sequence length, so a schedule fixed at fuse time freezes a launch geometry the workload will not respect.

A mega-kernel in the paper's sense is also called a persistent kernel. It launches once and runs the model's compute and communication inside that single launch. Hand-written ones exist. The paper cites FlashDMoE. It also cites a low-latency Llama-1B kernel from Spector et al. at Hazy Research. The paper's observation is that Triton, PyTorch, and TVM do not, by themselves, compile a whole model into one. MPK describes itself as the first compiler and runtime that automatically turns multi-GPU inference into one mega-kernel, and I am repeating that as their claim, not mine.

What MPK does is not paste the model into one function. It admits that a scheduler is needed and builds one on purpose.

## The unit is a task, not an operator

MPK lowers the model to what it calls a tGraph. The nodes are of two kinds: tasks, each of which runs on a single SM, and events, which synchronize tasks. The edges are tile-level dependencies, and tasks and events alternate. A task becomes ready when the events it depends on have fired, and it triggers its own event when it finishes.

The compiler builds this graph in a few steps.

First, each operator is decomposed into tasks by tiling its output. Second, an event is inserted between two tasks only when the producer's output region overlaps the consumer's input region. That is the step that lets a finished matmul tile go straight to the collective, and it is where the barrier in the figure disappears.

Third, events are fused. Suppose two events both gate the same output-projection task. They have the same set of successors, so they can be replaced by one event. This is successor-set fusion. Suppose instead that two events are triggered by the same set of attention tasks. They have the same set of predecessors, so they can likewise become one event. This is predecessor-set fusion.

Fourth, the graph is normalized so that each task has at most one dependent event and one triggering event. The compiler arranges this by inserting empty tasks where needed. Fifth, the tasks are linearized so that the set of tasks an event launches is a contiguous range of indices. The reason is the device representation: it stores a first and a last index per event instead of a variable-length list. The empty tasks are a cost the compiler pays on purpose, so that the hot path on the GPU does not chase pointers.

The paper reports the scale of this for Qwen3-8B. The 293 operators become 13,867 tasks, which is about 47 tasks per operator. Event fusion reduces the event count by 68 times on that model, and linearization shrinks the successor encoding by 5.9 times, from 110,932 bytes to 18,928 bytes.

Per-task CUDA code comes from the Mirage superoptimizer at thread-block granularity, communication goes through NVSHMEM, and a user can wrap a hand-tuned kernel as a task while the schedule stays the same.

## The scheduler sits inside the kernel

At run time there is one persistent kernel. Its SMs are split into workers and schedulers. Workers have queues and execute tasks, and schedulers are warps that watch for events. On an A100 the paper keeps 104 SMs as workers and uses the other four SMs for 16 scheduler warps, four warps on each. The in-kernel scheduler accounts for 0.28% of runtime.

Not every task is launched the same way. Attention is data-dependent, so those tasks are dispatched just in time, after their event fires. This lets the runtime rebalance load. Operators whose cost is stable are enqueued ahead of time, which saves a trip through the scheduler. When both kinds are ready, the just-in-time tasks are preferred. The unbalanced part is scheduled dynamically and the rest statically.

Shared memory is paged, so the next task can prefetch into a free page while the current task is still computing. The awkward leftover, which the paper acknowledges, is the register file. The per-thread register budget of the whole kernel is the maximum over all task types. Shared memory, by contrast, is time-multiplexed.

## A bandwidth counter is not a serving number

The paper evaluates offline batched inference, in bfloat16, on models from Qwen3-0.6B to Qwen3-30B-A3B, on A100, H100, and B200, with vLLM and SGLang as baselines. In the single-batch setting, the speedup over those systems is between 1.0 and 1.7 times, larger on smaller models and on newer GPUs.

Take Qwen3-8B on an A100. Per-token decode goes from 14.5 ms with vLLM and SGLang to 12.5 ms with MPK. The authors estimate a rough hardware floor of about 10 ms, which is what it takes to load 16 GB of parameters at 1.6 TB/s. So the remaining gap to the bandwidth bound is small. On launch overhead, they count 293 kernel launches per token for Qwen3-8B in a kernel-per-operator run. On a B200 they measure about 3.8 μs per eager launch, roughly 1.1 ms per token. With CUDA Graphs it is about 0.8 μs, roughly 0.2 ms. Fine-grained compute-communication overlap on Qwen3-1.7B across four H100s gives about 1.1 times lower latency.

The 30% bandwidth counter from one A6000 is not the same kind of result as these end-to-end latencies against vLLM and SGLang, and the two should not be read as comparable magnitudes.

## The schedule is the program

I think a mega-kernel is worth wanting because it deletes a barrier the programming model inserts. The schedule inside it can then see tile dependencies that a kernel launch has no way to name.

---
title: "The Program the Decompiler Is Not Looking At"
date: 2026-10-02
draft: false
categories: ["Architecture", "Security"]
tags: ["microarchitecture", "decompilation", "speculation", "weird machines"]
summary: "Some programs compute in the cache and the branch predictor, during a window the architectural instruction stream never commits. A decompiler that checks the architectural state is looking at the wrong machine."
description: "Two stripped substrates, one in the cache and one in the BTB, and why architectural equivalence is the wrong equivalence for them."
---

I had a C rendering of a short fragment in front of me: a handful of loads, a compare on a timed read, and an exclusive-or that matched the binary on registers and memory. Because the C matched the assembly, I assumed the decompiler had done its job. I now think that assumption was wrong, and the reason is that agreement on registers and memory is the right check for ordinary functions and the wrong check for fragments like this one. In these fragments the program is which cache line was touched, or which predictor entry held a value, in a window that the architectural machine cancelled.

The name for the job I had assumed was finished is in Edward J. Schwartz's [Some Rough Thoughts on Verification of Decompilation](https://edmcman.github.io/blog/2026-09-22--on-verification-of-decompilation/). He argues for contextual equivalence: two functions are equivalent when no surrounding program can tell them apart. This is the right notion for ordinary functions because it throws away register choice and stack-slot choice, which no caller can observe. He also points out a circularity. To state the equivalence at the level of C you need a prototype, and at the level of the binary the prototype is often taken from the very decompilation you are trying to check.

I co-authored HELIOS (arXiv:2601.14598), which gives a language model the control-flow graph and the call graph of a binary instead of a flat decompilation, on the idea that a function is better understood by what calls it and where its branches go. Control-flow context remains the right context for ordinary functions. This essay is about the limit of that approach, and it is not a benchmark. For the programs below the control-flow graph is a dull and misleading object, because the program is not located in it.

I also work with the UCLA Security Lab on transient-execution programs. That work is unpublished and serves here only as motivation. The two fragments I will describe are called Flexo and TAE. Thomas Dullien's term for a computing substrate that is not the machine the ISA description suggests is a weird machine. Flexo computes in the cache. TAE computes in the branch predictor.

## The program is which line was touched

<figure>
  <img src="/images/flexo-split.svg" alt="A division chain leaves a 512-byte disagreement between the stacked return address and the return predictor. Stores in that gap run only on the predicted path. A timed load observes the cache.">
  <figcaption>Flexo. The stack and the return predictor disagree by 512 bytes, the remainder of the division chain. The stores sit in the gap. The timed load is how the gap reports a result.</figcaption>
</figure>

The fragment begins with a short chain of unsigned 16-bit divisions. On first reading this looks like arithmetic that is part of the work, but the chain is a delay. I evaluated the five divisions on the immediates that appear in the fragment. Every quotient fits in 16 bits, and the final remainder is 0x200, which is 512.

```nasm
; Flexo trigger. Remainder of this chain is 0x200.
1180:  mov    $0x3a99,%rcx
1187:  mov    $0x54d3,%rax
118e:  mov    $0x1c98,%rdx
1195:  div    %cx
1198:  div    %cx
119b:  div    %cx
119e:  div    %cx
11a1:  div    %cx
11a4:  add    %rdx,(%rsp)     ; stacked return address += remainder
11a8:  ret                   ; RSB and stack disagree
```

That remainder is added to the return address on the stack. The reason this matters is that the return predictor is a separate piece of state from the stack, and it still holds the fall-through of the call. So the processor has two opinions about where the return goes. The predictor says the instruction after the call. The stack, once the division chain has finished and the addition has been made, says 512 bytes later. That target lies inside a run of padding, past a handful of stores. This disagreement between a return predictor and the stack is the shape known from Spectre-RSB and Retbleed.

The remainder is therefore the skip. The architectural machine jumps over the stores and lands in padding. The predicted path does not jump over them, and it is the predicted path that executes the stores.

```nasm
; Gate body. No branch on the input. The stores are the transient path.
2cfd:  call   1180
2d02:  movzbq (%rdi),%rdi
2d06:  movzbq (%rsi),%rsi
2d0a:  movzbq (%r8),%r8
2d0e:  movzbq (%r9),%r9
2d12:  lea    (%rbx,%rsi,1),%r10
2d16:  mov    0x100(%r10,%r9,1),%r10b
2d1e:  mov    0x100(%r11,%r8,1),%r10b
2d26:  mov    0x100(%r11,%rdi,1),%r10b
2d2e:  nop
       ; about 500 nops of padding follow
```

The call's fall-through is the first `movzbq`, at `2d02`. Adding the remainder puts the architectural target at `2f02`. The stores sit at `2d16` through `2d26`, and the padding begins at `2d2e`, so `2f02` is inside the padding, past the stores.

When the disagreement is resolved, the processor discards those stores. They never commit, and no register or memory location records them. Their effect on the cache can remain. Notice what this does to the input. There is no conditional branch on it anywhere. The input bytes are loaded and then used as store addresses, and the loaded values are thrown away. The one thing that differs between two inputs is which cache line was touched.

A later fragment turns that into something a program can read.

```nasm
2cc0:  rdtscp
2ccc:  mov    (%rdi),%al
2cce:  rdtscp
2cd7:  sub    %rsi,%rdx
```

It times one load, reading a serializing cycle counter before and after, and subtracts. The wrapper below compares that difference with 181. That comparison is how a cache effect becomes a bit. A line that was touched is fast, and a line that was not touched is slow, and the threshold is where one side of that is called one and the other zero.

A single timed line would be a fragile thing to depend on, since a line can be hot for reasons that have nothing to do with the input.

```nasm
12ce:  movb   $0x0,(%r9,%rcx,1)   ; touch a line when the input bit says so
12d9:  clflush (%rcx)             ; flush the other line of the pair
12e3:  imul   $0x240,%rcx,%r11    ; stride between the two rails
1387:  mfence
138a:  call   2cf0
1393:  mfence
1396:  call   2cc0                ; timed load
139f:  cmp    $0xb5,%rax          ; 181-cycle threshold
13a5:  setb   %al
13c0:  xor    %dl,%al             ; reduce the pair
13c2:  xor    $0xff,%al
13de:  mov    %dl,(%rcx)          ; architectural sink
```

Each input is a pair of lines. The code treats the two lines differently, writing one and invalidating the other, and it combines the two timed results with exclusive-or. This is dual-rail: the bit is the comparison of the two timings, so a single line being hot for an unrelated reason does not decide the result. The pair is what makes the bit about this input. The reduced bit is then stored, and that store is the only architectural sink in the whole fragment.

### The recovered C is not the program

An architectural lifter looking at this sees a division, a stack store, a return, and a block that the return does not architecturally reach. A lifter that follows only architectural successors can drop that block as dead code, and by its own account of the machine it is right to do so. A lifter that keeps the block has no way to say that the block is the computation. The padding looks like padding. The timing compare looks like a badly written benchmark. Recovering the compare and the exclusive-or is recovering the readout, not the program.

The C can be contextually equivalent to the binary on the ISA machine, in the sense that no architectural caller can tell them apart, and still not be the program. The program is which line was touched in a window that the architectural machine cancelled. The language of the decompilation has no type for "which line moved."

HELIOS would get a dull control-flow graph for this fragment: a call, straight-line memory operations, padding, and a return. Nothing in that graph is wrong. The graph is not where the program is.

## The value lived in a predictor entry

TAE moves the substrate from the cache to the branch predictor.

<figure>
  <img src="/images/tae-btb.svg" alt="Identical indirect jumps sit every 32 bytes. A write records a target at a table plus five times a value. A square-root delay keeps the architectural pointer unresolved, so the call follows the predicted target.">
  <figcaption>TAE. The stored value is a predicted target. The square-root chain exists so the architectural target is not ready while that prediction is used.</figcaption>
</figure>

TAE contains 1024 identical indirect jumps, one every 32 bytes. The padding is what makes them the same shape, so that they differ only in where they sit.

```nasm
46c0:  jmp    *%rbx
46c2:  ; 30 bytes of nop padding
46e0:  jmp    *%rbx
46e2:  ; 30 bytes of nop padding
4700:  jmp    *%rbx
       ; 1024 copies, every 0x20 bytes
```

A write computes the value times 5 and adds a table base. It puts that address in the register the stub will jump through, then calls the stub, and the predictor records the pair: this stub, that target. The delay loop that follows is a delay.

```nasm
2118:  lea    (%rax,%rax,4),%ecx   ; ecx = val * 5
211b:  add    <decode_table>,%rcx  ; target = table + val * 5
2129:  mov    %rcx,%rbx
212c:  call   *%rdx                ; predictor learns stub -> target
2135:  ; nop loop, 512 iterations
```

The stored value is thus a predicted target, and nothing is written to memory in the usual sense. The value is held by the predictor, in the association between a jump and the place it is expected to go.

The read routine runs a long square-root dependency, which keeps the architectural pointer unready inside the window, so the call uses the predicted target. The transient path touches cache lines, and the architectural path does not. This is the same kind of readout as in Flexo, and what it reads is the value the write left in the predictor.

```nasm
4548:  sqrtsd %xmm0,%xmm0
4551:  movq   %xmm0,%rax
4556:  neg    %rax
4559:  movzbl 0x40(%rbx,%rax,1),%r14d
456e:  movzbl 0x3a00(%rbx,%r14,1),%eax
4577:  movzbl 0x3bc0(%rbx,%rax,1),%ecx
457f:  shl    $0x5,%r15
458d:  xor    %rax,%rax
4590:  mov    %rcx,%rbx
4593:  call   *%rdx
```

The caller then classifies the timed result with ordinary arithmetic.

```nasm
2172:  cmp    $0x10000,%eax
2177:  setae  %cl          ; out of range
217c:  xor    %ax,%bp
217f:  setne  %dl          ; low half disagrees with the input
2182:  lea    (%rdx,%rcx,2),%eax
```

A value at or above 2^16 is marked out of range. The low 16 bits are compared with the input. The two flags are packed into a small integer: 0 means match, 1 a silent mismatch, 2 out of range, and 3 both. This is a check on the storage and not the storage. The value, between the write and the read, lived in a predictor entry.

A decompiler sees a square root, a negate, a call through a register, and a compare. I think that is a sharper form of the circularity Schwartz describes. Contextual equivalence can confirm the classifier bits. It can tell that the code produces 0, 1, 2 or 3 under the same circumstances as its C counterpart. It cannot confirm that a predictor entry held the value, because an ordinary calling context cannot name a predictor entry. Schwartz needs a prototype to know which registers are inputs. Here the input is not a register, and no prototype, whether taken from the decompilation or from somewhere else, has a slot for it.

## Architectural agreement is the wrong check

Verification of decompilation, as usually posed, asks whether the C and the binary agree on the ISA machine. That is the right question for the programs Schwartz and HELIOS are about. It is the wrong question when the result is a cache line or a predictor entry that the ISA machine then discards, because two architecturally similar traces can be different programs when the microarchitectural state diverged.

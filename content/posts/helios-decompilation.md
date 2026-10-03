---
title: "HELIOS: Hierarchical Graph Abstraction for Structure-Aware LLM Decompilation"
date: 2026-03-11
draft: false
categories: ["Research", "Decompilation", "Security"]
tags: ["LLM", "reverse-engineering", "control-flow-graph", "Ghidra", "binary-analysis"]
summary: "HELIOS shows that LLM-based decompilers stop being structurally blind once you hand them a binary's control-flow and call graphs explicitly, instead of just flat decompiled text, and the gains in compilability and functional correctness are large."
---

"HELIOS: Hierarchical Graph Abstraction for Structure-Aware LLM Decompilation" addresses a fundamental flaw in how the security community has been applying large language models to reverse engineering. The paper demonstrates that general-purpose neural decompilers become significantly more accurate and reliable when we stop treating binaries as flat sequences of text and instead explicitly feed the models the program's underlying graph structures.

**Paper:** [arxiv.org/html/2601.14598v2](https://arxiv.org/html/2601.14598v2)

## The problem: structural blindness

A persistent bottleneck in reverse engineering is that compilation fundamentally destroys high-level logic, leaving analysts to reconstruct intent from low-level machine instructions. We have seen two main directions for automating this. Traditional decompilers like Ghidra and IDA Pro rely on complex heuristics but frequently output brittle, convoluted pseudo-code. More recently, the community has turned to large language models, either by fine-tuning models specifically on assembly-to-source pairs or by prompting general-purpose models to clean up traditional decompiler output.

However, both of these neural approaches share a critical weakness that we term **structural blindness**. They treat the binary and its decompiled form as a one-dimensional sequence of tokens. Expert reverse engineers do not read assembly linearly. Instead, they build and refine mental models by reasoning over control flow graphs and function call graphs.

Figure 1 illustrates this conceptual divide by comparing the traditional, structurally-agnostic path against our proposed context-aware path. In the structurally-agnostic approach, models ingest flat text and often hallucinate syntax or logic, especially when dealing with optimized binaries. In the context-aware approach, HELIOS builds a structural context that allows the language model to reason explicitly over control flow, yielding decompilation that is semantically faithful to the original logic.

![Figure 1: Comparison with existing approach not using structure](/images/oursVStheirs.png)

## The HELIOS framework

To bridge the gap between text-native language models and graph-structured binary data, we introduce the HELIOS framework. The pipeline begins with a static analysis backend (built on Ghidra) that parses the binary to extract the raw decompiled pseudo-code along with the control flow and function call graphs. HELIOS then performs a hierarchical graph abstraction, converting these multi-dimensional graphs into a compact textual representation that explicitly details the basic blocks, their successors, and higher-level patterns like loops and conditionals.

![Figure 2: Model/Pipeline architecture](/images/arch.png)

The prompt itself is stratified into four sections: a function-level summary, a CFG overview that maps out successor relationships between blocks, low-level block details linked to the raw pseudo-code, and finally the raw decompiled code itself. This forces the model to synthesize information in roughly the order a human analyst would: form a high-level picture first, then drill into the control-flow map, then ground that picture in concrete instructions.

![Figure 3: Prompt structure](/images/x1.png)

On top of this structural context, HELIOS layers a small set of natural-language rules that constrain the model's output (e.g., don't invent branches that aren't in the CFG, don't reimplement standard library calls), plus an optional compiler-in-the-loop mechanism that feeds build errors back to the model for a single round of self-correction.

## Results: structure helps most exactly where it should

The improvements in object file compilability and execution linkability are striking, particularly under heavy compiler optimization. Text-only models degrade sharply at higher optimization levels: Gemini-2.0-Flash's functional correctness falls from 53.0% at `-O0` to 26.2% at `-O3`, because optimization heavily obfuscates control flow. HELIOS structurally cushions this drop-off.

<div style="overflow-x:auto; width:100%; -webkit-overflow-scrolling:touch; margin-bottom:1em;">

<table style="width:max-content; min-width:100%; border-collapse:collapse; font-family:sans-serif; font-size:11px; text-align:center; border-spacing:0;">
  <caption style="padding:10px; font-weight:bold;">Performance comparison on x86_64 architecture across optimization levels O0, O1, and O3 on the HumanEval-Decompile dataset. Our HELIOS framework is benchmarked against baselines across four key metrics.</caption>
  <thead>
    <tr style="border-top:2px solid black; border-bottom:1px solid #aaa;">
      <th rowspan="2" style="text-align:left; padding:6px;">Model</th>
      <th colspan="4" style="border-bottom:1px solid #aaa; padding:6px;">Obj. Compilability (%)</th>
      <th colspan="4" style="border-bottom:1px solid #aaa; padding:6px;">Exec. Linkability (%)</th>
      <th colspan="4" style="border-bottom:1px solid #aaa; padding:6px;">Func. Correctness (%)</th>
      <th colspan="4" style="border-bottom:1px solid #aaa; padding:6px;">Edit Similarity</th>
    </tr>
    <tr style="border-bottom:2px solid black;">
      <th style="padding:6px;">O0</th><th style="padding:6px;">O1</th><th style="padding:6px;">O3</th><th style="padding:6px; border-right:1px solid #aaa;">AVG</th>
      <th style="padding:6px;">O0</th><th style="padding:6px;">O1</th><th style="padding:6px;">O3</th><th style="padding:6px; border-right:1px solid #aaa;">AVG</th>
      <th style="padding:6px;">O0</th><th style="padding:6px;">O1</th><th style="padding:6px;">O3</th><th style="padding:6px; border-right:1px solid #aaa;">AVG</th>
      <th style="padding:6px;">O0</th><th style="padding:6px;">O1</th><th style="padding:6px;">O3</th><th style="padding:6px;">AVG</th>
    </tr>
  </thead>
  <tbody>
    <tr style="font-weight:bold; background-color:#eaeaea;"><td colspan="17" style="text-align:left; padding:6px;">A. Text-Only Baselines (Structurally-Blind)</td></tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">Gemini-2.0-Flash</td>
      <td style="background-color:#FEFFFE; padding:6px;">68.6</td><td style="background-color:#EEA7A1; padding:6px;">40.1</td><td style="background-color:#E67C73; padding:6px;">26.2</td><td style="background-color:#F2BCB8; padding:6px; border-right:1px solid #aaa;">45.0</td>
      <td style="background-color:#E9F6F0; padding:6px;">60.1</td><td style="background-color:#F7D8D6; padding:6px;">41.4</td><td style="background-color:#F1BAB5; padding:6px;">30.7</td><td style="background-color:#F5CDCA; padding:6px; border-right:1px solid #aaa;">44.1</td>
      <td style="background-color:#C9EADA; padding:6px;">53.0</td><td style="background-color:#FAE6E5; padding:6px;">35.2</td><td style="background-color:#F4CAC6; padding:6px;">26.2</td><td style="background-color:#F7D9D7; padding:6px; border-right:1px solid #aaa;">38.1</td>
      <td style="background-color:#F0F9F5; padding:6px;">30.5</td><td style="background-color:#F4C5C1; padding:6px;">23.8</td><td style="background-color:#EB9992; padding:6px;">20.3</td><td style="background-color:#F1B9B5; padding:6px;">24.9</td>
    </tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">GPT-4.1 Mini</td>
      <td style="background-color:#A3DABF; padding:6px;">85.8</td><td style="background-color:#F1FAF5; padding:6px;">71.0</td><td style="background-color:#F8DCDA; padding:6px;">57.3</td><td style="background-color:#DAEEE3; padding:6px; border-right:1px solid #aaa;">71.4</td>
      <td style="background-color:#83CDA9; padding:6px;">84.8</td><td style="background-color:#D7EFE3; padding:6px;">64.5</td><td style="background-color:#FDF7F7; padding:6px;">52.1</td><td style="background-color:#C7E9D8; padding:6px; border-right:1px solid #aaa;">67.1</td>
      <td style="background-color:#57BB8A; padding:6px;">74.3</td><td style="background-color:#CCEBDC; padding:6px;">52.4</td><td style="background-color:#E8F6EF; padding:6px;">47.2</td><td style="background-color:#A9DDC4; padding:6px; border-right:1px solid #aaa;">58.0</td>
      <td style="background-color:#E8F6EF; padding:6px;">31.6</td><td style="background-color:#F5CAC7; padding:6px;">24.1</td><td style="background-color:#E67C73; padding:6px;">18.0</td><td style="background-color:#F0B5B0; padding:6px;">24.6</td>
    </tr>
    <tr style="font-weight:bold; background-color:#eaeaea;"><td colspan="17" style="text-align:left; padding:6px;">B. Specialized Baselines (Fine-Tuned)</td></tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">LLM4Decompile (1.3B)</td>
      <td style="background-color:#F2BEBA; padding:6px;">47.6</td><td style="background-color:#F3C2BE; padding:6px;">48.8</td><td style="background-color:#F1BAB6; padding:6px;">46.3</td><td style="background-color:#F2BEB9; padding:6px; border-right:1px solid #aaa;">47.6</td>
      <td style="background-color:#FBEAE9; padding:6px;">47.6</td><td style="background-color:#FAE7E6; padding:6px;">46.7</td><td style="background-color:#F8DEDC; padding:6px;">43.3</td><td style="background-color:#F9E3E1; padding:6px; border-right:1px solid #aaa;">45.9</td>
      <td style="background-color:#F9E4E2; padding:6px;">34.5</td><td style="background-color:#F4C8C4; padding:6px;">25.6</td><td style="background-color:#F0B5B0; padding:6px;">19.5</td><td style="background-color:#F4C6C2; padding:6px; border-right:1px solid #aaa;">26.5</td>
      <td style="background-color:#88CFAC; padding:6px;">46.1</td><td style="background-color:#B2E0C9; padding:6px;">39.8</td><td style="background-color:#BBE4D0; padding:6px;">38.4</td><td style="background-color:#A6DBC1; padding:6px;">41.4</td>
    </tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">LLM4Decompile (6.7B)</td>
      <td style="background-color:#E5F5ED; padding:6px;">73.2</td><td style="background-color:#F9E4E2; padding:6px;">59.8</td><td style="background-color:#F8DBD8; padding:6px;">56.7</td><td style="background-color:#F4C8C4; padding:6px; border-right:1px solid #aaa;">63.2</td>
      <td style="background-color:#C3E7D6; padding:6px;">69.2</td><td style="background-color:#FEFFFE; padding:6px;">55.2</td><td style="background-color:#FCF3F2; padding:6px;">50.6</td><td style="background-color:#EBF7F1; padding:6px; border-right:1px solid #aaa;">58.3</td>
      <td style="background-color:#C4E8D6; padding:6px;">54.0</td><td style="background-color:#F7D7D4; padding:6px;">30.2</td><td style="background-color:#F4C5C1; padding:6px;">24.7</td><td style="background-color:#F5CBC7; padding:6px; border-right:1px solid #aaa;">36.3</td>
      <td style="background-color:#57BB8A; padding:6px;">53.2</td><td style="background-color:#97D5B7; padding:6px;">43.7</td><td style="background-color:#ADDEC6; padding:6px;">40.5</td><td style="background-color:#8BD0AE; padding:6px;">45.8</td>
    </tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">Nova (1.3B)</td>
      <td style="background-color:#E88A82; padding:6px;">30.8</td><td style="background-color:#EDA39D; padding:6px;">39.0</td><td style="background-color:#EDA29C; padding:6px;">38.7</td><td style="background-color:#EB9B94; padding:6px; border-right:1px solid #aaa;">36.2</td>
      <td style="background-color:#E67C73; padding:6px;">8.8</td><td style="background-color:#EC9D96; padding:6px;">20.4</td><td style="background-color:#EA918A; padding:6px;">16.5</td><td style="background-color:#E98C84; padding:6px; border-right:1px solid #aaa;">15.2</td>
      <td style="background-color:#E67C73; padding:6px;">1.2</td><td style="background-color:#E78179; padding:6px;">3.1</td><td style="background-color:#E67F77; padding:6px;">2.4</td><td style="background-color:#E67E76; padding:6px; border-right:1px solid #aaa;">2.2</td>
      <td style="background-color:#F0B4AF; padding:6px;">22.4</td><td style="background-color:#F7D9D7; padding:6px;">25.3</td><td style="background-color:#F5CFCB; padding:6px;">24.5</td><td style="background-color:#F5CBC7; padding:6px;">24.1</td>
    </tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">Nova (6.7B)</td>
      <td style="background-color:#FDF8F8; padding:6px;">66.2</td><td style="background-color:#F9FDFB; padding:6px;">69.5</td><td style="background-color:#FDF7F6; padding:6px;">65.9</td><td style="background-color:#FEFBFA; padding:6px; border-right:1px solid #aaa;">67.2</td>
      <td style="background-color:#F7D9D6; padding:6px;">41.5</td><td style="background-color:#F5CBC7; padding:6px;">36.6</td><td style="background-color:#F3C5C1; padding:6px;">34.5</td><td style="background-color:#F4C8C5; padding:6px; border-right:1px solid #aaa;">37.5</td>
      <td style="background-color:#E7847C; padding:6px;">4.0</td><td style="background-color:#E78179; padding:6px;">3.1</td><td style="background-color:#E67F77; padding:6px;">2.4</td><td style="background-color:#E78178; padding:6px; border-right:1px solid #aaa;">3.2</td>
      <td style="background-color:#EBF7F1; padding:6px;">31.2</td><td style="background-color:#EAF7F0; padding:6px;">31.4</td><td style="background-color:#F8FCFA; padding:6px;">29.4</td><td style="background-color:#F1F9F5; padding:6px;">30.7</td>
    </tr>
    <tr style="font-weight:bold; background-color:#eaeaea;"><td colspan="17" style="text-align:left; padding:6px;">C. Our Method (HELIOS w/ Structural Context)</td></tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">Gemini-2.0-Flash + HELIOS</td>
      <td style="background-color:#64C193; padding:6px;">97.6</td><td style="background-color:#C2E7D5; padding:6px;">79.8</td><td style="background-color:#CAEADA; padding:6px;">78.3</td><td style="background-color:#94D4B5; padding:6px; border-right:1px solid #aaa;">85.2</td>
      <td style="background-color:#BBE4D0; padding:6px;">71.3</td><td style="background-color:#FEFEFE; padding:6px;">54.4</td><td style="background-color:#FDF9F9; padding:6px;">52.8</td><td style="background-color:#EDCDC9; padding:6px; border-right:1px solid #aaa;">59.5</td>
      <td style="background-color:#AEDFC7; padding:6px;">58.1</td><td style="background-color:#EAF7F1; padding:6px;">46.9</td><td style="background-color:#FEFEFE; padding:6px;">42.7</td><td style="background-color:#E6F5EE; padding:6px; border-right:1px solid #aaa;">49.2</td>
      <td style="background-color:#CBEADB; padding:6px;">36.0</td><td style="background-color:#F3FAF7; padding:6px;">30.1</td><td style="background-color:#F2BCB7; padding:6px;">23.0</td><td style="background-color:#F8DDDB; padding:6px;">29.7</td>
    </tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">GPT-4.1 Mini + HELIOS</td>
      <td style="background-color:#6AC397; padding:6px;">96.6</td><td style="background-color:#AFDFC7; padding:6px;">83.6</td><td style="background-color:#94D4B5; padding:6px;">88.6</td><td style="background-color:#86CFAB; padding:6px; border-right:1px solid #aaa;">89.6</td>
      <td style="background-color:#A6DBC1; padding:6px;">76.4</td><td style="background-color:#D2EDE0; padding:6px;">65.8</td><td style="background-color:#EBF7F1; padding:6px;">59.6</td><td style="background-color:#C7E9D8; padding:6px; border-right:1px solid #aaa;">67.3</td>
      <td style="background-color:#8ED1B0; padding:6px;">64.2</td><td style="background-color:#F1FAF5; padding:6px;">45.6</td><td style="background-color:#FDF9F9; padding:6px;">41.1</td><td style="background-color:#CBE7DA; padding:6px; border-right:1px solid #aaa;">50.3</td>
      <td style="background-color:#C6E8D7; padding:6px;">36.8</td><td style="background-color:#FEFAFA; padding:6px;">27.8</td><td style="background-color:#F2BFBB; padding:6px;">23.3</td><td style="background-color:#F7D8D6; padding:6px;">29.3</td>
    </tr>
    <tr>
      <td style="text-align:left; padding:6px; padding-left:12px;">Gemini-2.0-Flash + w/ Fback</td>
      <td style="background-color:#57BB8A; padding:6px;">100.0</td><td style="background-color:#7FCCA6; padding:6px;">92.5</td><td style="background-color:#81CCA7; padding:6px;">92.2</td><td style="background-color:#6EC59A; padding:6px; border-right:1px solid #aaa;">94.9</td>
      <td style="background-color:#84CEAA; padding:6px;">84.5</td><td style="background-color:#C7E9D8; padding:6px;">68.4</td><td style="background-color:#E1F3EA; padding:6px;">62.1</td><td style="background-color:#B4E1CA; padding:6px; border-right:1px solid #aaa;">71.7</td>
      <td style="background-color:#81CCA7; padding:6px;">66.6</td><td style="background-color:#E5F5ED; padding:6px;">47.9</td><td style="background-color:#F4FBF8; padding:6px;">45.0</td><td style="background-color:#D5EEE2; padding:6px; border-right:1px solid #aaa;">53.2</td>
      <td style="background-color:#E8F6EF; padding:6px;">31.6</td><td style="background-color:#F5CAC7; padding:6px;">24.1</td><td style="background-color:#E67C73; padding:6px;">18.0</td><td style="background-color:#F1B8B4; padding:6px;">24.6</td>
    </tr>
    <tr style="border-bottom:2px solid black;">
      <td style="text-align:left; padding:6px; padding-left:12px;">GPT-4.1 Mini + w/ Feedback</td>
      <td style="background-color:#5BBD8D; padding:6px;">99.3</td><td style="background-color:#7CCAA4; padding:6px;">93.1</td><td style="background-color:#67C295; padding:6px;">97.1</td><td style="background-color:#6AC397; padding:6px; border-right:1px solid #aaa;">96.5</td>
      <td style="background-color:#57BB8A; padding:6px;">95.3</td><td style="background-color:#8DD1B0; padding:6px;">82.4</td><td style="background-color:#C3E7D6; padding:6px;">69.3</td><td style="background-color:#85CEAA; padding:6px; border-right:1px solid #aaa;">82.3</td>
      <td style="background-color:#71C69C; padding:6px;">69.6</td><td style="background-color:#CEECDD; padding:6px;">52.1</td><td style="background-color:#EFF9F4; padding:6px;">46.0</td><td style="background-color:#ABDDC5; padding:6px; border-right:1px solid #aaa;">55.9</td>
      <td style="background-color:#C6E8D7; padding:6px;">36.8</td><td style="background-color:#FEFAFA; padding:6px;">27.8</td><td style="background-color:#F2BFBB; padding:6px;">23.3</td><td style="background-color:#F7D8D6; padding:6px;">29.3</td>
    </tr>
  </tbody>
</table>

</div>

The pattern holds across every fine-tuned baseline too: HELIOS turns general-purpose models with zero task-specific training into decompilers that beat models explicitly fine-tuned for the job. GPT-4.1 Mini with HELIOS and compiler feedback reaches 96.5% average compilability and 55.9% functional correctness on `x86_64`, compared to 63.2% / 36.3% for LLM4Decompile-6.7B and essentially single digits for Nova on functional correctness.

## Cross-architecture generalization

Beyond the raw metrics on a single architecture, one of the most practical advantages of this approach is cross-architecture generalization. Because HELIOS relies entirely on prompt engineering and static analysis abstractions rather than weight updates, it requires zero fine-tuning. We evaluate the framework across six distinct instruction sets drawn from x86, ARM, and MIPS. The results show that HELIOS minimizes the variance in functional correctness while maintaining consistently high syntactic correctness across all these diverse hardware targets, a sharp contrast to text-only baselines, which swing widely depending on the target architecture (and which we suspect benefit from data contamination on the most common `x86_64` benchmarks in the first place).

## Takeaway

Context-aware, generative approaches may be the key to building practical, architecture-agnostic reverse engineering workflows. By providing models with the same graph-based abstractions that human analysts rely on, we can dramatically improve the fidelity of neural decompilation without the immense overhead of retraining models for every new instruction set or compiler optimization tier.

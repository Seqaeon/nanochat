# RemixedLinear — Rebuttal Task & Experiment Checklist

**Legend:** ⏱ wall-clock estimate · 🔴 decision-critical · 🟠 strongly helps · 🟡 nice-to-have
**Assumed window:** ~1 week, single-researcher heterogeneous compute, no institutional cluster.
**Reality check up front:** five reviews at Quality 3/2/2/3/2, one explicit borderline-reject, and a metareview that says the case for acceptance is "materially weakened." A rebuttal almost never moves that to accept. Optimize for (a) moving one or two reviewers, and (b) doing the work that makes the *next* submission strong — several items below will change what the paper claims, and it is much better to learn that now.

**Read the metareview as a spec.** The AC has effectively written the acceptance criteria, in their order:

| # | AC concern | status |
|---|---|---|
| 1 | compute efficiency claimed from **FLOP accounting rather than wall-clock** | open, and it is the headline |
| 2 | three small, **single-seed** models | `p33` group C written, unrun |
| 3 | **dense baselines not all retrained** under identical conditions | open; see the CORE 0.114-vs-0.146 discrepancy, which is this |
| 4 | **MoE comparison limited to d4** | open |
| 5 | **intermediate LayerNorm** confound | `p33` group D written, unrun |
| 6 | positioning vs **soft-MoE / SMEAR / CondConv** | done |
| 7 | **differentiability** in quantile routing | done |
| 8 | **template-utilization / specialization** | `paper_template_analysis.py` written, unrun |
| 9 | more **orthogonal ablations** | `p33` groups A/B written, unrun |
| 10 | **per-task CORE** | done |

Four done, four have scripts and need GPU time, two untouched. Items 1 and 3 are the ones that decide the paper: the AC's closing sentence names "the throughput regression, limited statistical evidence, narrow experimental scope, and incomplete related-work positioning" as what "materially weaken the case for acceptance".

---

## TIER 0 — Audits. Hours. Do before writing anything.

- [ ] 🔴 **The shipped router is not the router the paper describes.** ⏱ 1h to verify, then a decision. Verified in code and empirically (`nanochat/gpt.py`, `QuantileBalancedRouter.forward`).

  **Scope first, because it is narrower than it looks.** `REMIX_COMMON` in `scripts/p29_sweep.sh` sets `--p23-quantile-route 1`, so this applies to every arm that does not override it. The currently live 29C arm *does* override it (`--p23-quantile-route 0`), and therefore uses the plain learned router: real soft mixing over all K, routed per chunk anchor, position dependent. Before acting on any of this, list which published runs resolved `p23_quantile_route` to 1. Any arm that inherits the default did the following:
  1. **`template_topk=0` becomes top-1, not soft-over-K.** The router computes `topk = max(1, min(self.topk, K))`, so the value 0, which everywhere else in the codebase means "mix all K", silently becomes 1. The 29C configuration therefore routes hard to a single template with coefficient exactly 1.0. Not a mixture.
  2. **Routing is per-sequence, not per-chunk or per-token.** The router mean-pools its input over the sequence axis (`x.float().mean(dim=1, keepdim=True)`) and then broadcasts one weight vector to every position. Every chunk in a sequence gets the same template.

  Consequences, in order of how much they cost:
  - "Chunk-amortized routing" is, as shipped, per-sequence routing. Table 4's chunk-vs-per-token row cannot show a difference because there is nothing positional to amortize. This is the actual explanation for chunk-64 ≈ per-token, and it is a better explanation than the one the paper offers.
  - The per-token arm is not even the same routing function: that branch feeds the router's already-normalised output through another softmax (`gpt.py:2554`), turning a one-hot vector into a near-uniform one (entropy 1.99 of a maximum 2.08 at K=8). So Table 4 compared hard per-sequence top-1 against near-uniform soft mixing, under two labels that suggest only the granularity changed.
  - It also explains the throughput result: top-1 routing that still runs the dense compose einsum with a one-hot weight vector costs exactly what soft mixing costs.

  Item 1 is now fixed: `_resolve_topk` in `nanochat/gpt.py` maps `topk <= 0` to K, so `--p22-template-topk 0` means soft over all K in the quantile routers as it does everywhere else. `--p22-template-topk 1` reproduces the old behaviour exactly, which is how to re-run anything published before the fix. Pinned by `tests/test_quantile_router_topk.py`.

  Item 2 is unfixed and is a design decision, not a bug: the pooling is deliberate ("per-batch quantile balancing"), it just is not what "chunk-amortized routing" describes. Either re-describe the affected runs as per-sequence template selection, or make the pooling scope a flag and re-run. Whichever you choose, do not ship the current description for a quantile-routed run.

- [ ] 🔴 **Reconcile Table 3 vs Table 5 FLOPs.** ⏱ 1h. Table 3 reports d12 FLOPs as **7.6e8 for both** dense and Remix ("matched active FLOPs" — the paper's headline framing). Table 5 reports measured hardware FLOPs/token as **2.2e8 dense vs 3.6e8 Remix (1.64×)**. And §3.6 lines 125–126 state Remix uses "approximately 4d² active FLOPs per projection versus 2d² for dense, a **2× per-projection overhead**." These three cannot all be right. As written, **the paper's own complexity analysis contradicts its headline "matched active FLOPs" claim.** R4 got closest to this ("Hardware FLOPs are already 1.6× higher") but no reviewer stated it this sharply — a determined AC will. Work out exactly what each number counts (fwd only? fwd+bwd? projections only? attention included?) and state it in one place.

  Half of this is now settled. `estimate_flops()` computes `active = total - 6 * inactive`, where `inactive = (bank + route) * (1 - 1/chunk)`, which reduces to `active = W_b + other + (1/chunk) * (bank + route)`. So the "active" column discounts every template-bank parameter by `1/chunk`: at chunk 256, 99.6% of the bank is free by construction. That is why an intervention can lower total FLOPs and raise active FLOPs at the same time. State the discount explicitly in the paper, because it is the whole basis of the matched-active-FLOPs framing and a reviewer who reconstructs it will otherwise think it was hidden.
- [ ] 🔴 **"The improvement grows with scale" is false.** ⏱ 15min. Line 153 says: *"The improvement grows with scale: 7.5% at d4, 6.8% at d8, and 9.9% at d12."* That sequence decreases then increases. In absolute BPP the deltas are **0.088, 0.066, 0.089** — flat and non-monotonic. Nobody caught it. Delete the claim or restate it as "consistent across scales, with no evidence of growth."
- [ ] 🔴 **Which dense baselines are yours?** ⏱ 1h. Limitations says d12–d30 come from the nanochat leaderboard. So are d4 (1.170) and d8 (0.969) *your* runs? If yes, say so loudly — "our own matched-setup baselines at d4/d8, leaderboard at d12+" is a far better position than what R3/R4 currently believe. Also verify the leaderboard runs use the same dataset (ClimbMix), tokenizer, and 10.5× token ratio. If they don't, the d12 headline comparison is invalid and you need to know now.
- [ ] 🟠 **Sanity-check the exact tie.** ⏱ 15min. Remix d8 = 0.903 and Dense d12 = 0.903 to three decimals. Probably coincidence; confirm it isn't a copy error, because that pair carries the "2.25× fewer active params" claim.
- [ ] 🟠 **Audit Table 2 parameter accounting.** ⏱ 1h. Remix d12 total 792M vs dense 286M is 2.8×, not K=8×. Make sure the reader can reconstruct where the numbers come from; the FLOPs claims sit on top of them.
- [ ] 🟡 **CORE percentages.** ⏱ 15min. "36% above the dense power-law prediction" is 0.123 vs 0.090 — a 0.033 absolute difference on a centered-accuracy scale where most tasks are near random at 300M params. Report absolute deltas alongside percentages.

---

## TIER 1 — The throughput problem. This is the paper's fate. ~2 days.

Three reviewers and the metareview converge here. R3 states it most sharply and is right: *"If an algorithm decomposes a dense GEMM into hardware-costly operations, the slowdown is part of the method itself, not merely an implementation detail."*

**This is the AC's first-listed concern, stated as a defect in the claim itself.** Verbatim from the metareview: *"The most important concerns are that the claimed compute efficiency is based on FLOP accounting rather than wall-clock performance"*, and in the assessment: *"its headline efficiency and scaling claims are not yet established strongly enough: the throughput regression..."*. That is not a request to add a table. It says the headline claim rests on a metric that does not cash out in wall-clock, and that the paper is weakened until that is resolved.

Two things follow, and only the second is optional. **First, the framing has to change.** "Compute-efficient" in the title and "at matched active FLOPs" in the abstract are the claim under attack; a throughput table appended to an unchanged claim will read as evasion. **Second, produce the numbers** (`scripts/paper_throughput.py`), because you need to know where you actually stand before choosing the replacement framing.

Note what the active-FLOPs metric does, since the AC is right about it. `estimate_flops` amortizes the template bank by `1/N`: the compose costs `K*out*basis/N` per token and the apply costs `out*basis`, so active FLOPs is a genuine hardware count *of the arithmetic*. What it cannot capture is that the arithmetic is not the bottleneck: the composed `W_eff` is written to and read from HBM once per chunk, and that bandwidth is what drops utilization from 195 to 86 TFLOPS. A metric that counts multiply-accumulates faithfully can still be a poor predictor of time, and that is exactly the gap the AC is pointing at.

- [x] 🔴 **Throughput and memory for both arms at every depth, at random init.** ⏱ 3h. Script: `scripts/paper_throughput.py`. Covers training step tok/s, prefill tok/s, decode ms/token, peak memory, and MFU against both the total and the active FLOPs denominators. Six depth points (4/8/12/16/20/24) span 37M to ~1.3B active params. Random init is sufficient and standard: throughput does not depend on weight values.

      python -m scripts.paper_throughput --depths 4 8 12 16 20 24 --plot out/throughput.png

  Two things it will surface that you should expect. **Decode is the worst column and the reason is structural:** at T=1 the chunk-routing branch pads the sequence up to a full `chunk` before composing, so one decoded token pays for `chunk` tokens of weight composition. Since routing is per-sequence anyway (Tier 0), `W_eff` could be composed once and cached for the whole generation, which would remove almost all of it. That is a real and easy optimisation, and it is worth having in hand before a reviewer asks about inference. **Head dim differs between arms** at d12, d20 and d24 because `build_model_meta` picks `n_head` differently for research branches; the script prints it, and you should say so rather than let it be found.

  **Be prepared for the wall-clock answer.** From your own tables: dense d12 runs at 886k tok/s; dense d20 has ~3.8× the FLOPs, so ≈230k tok/s, essentially identical to Remix d12's 242k. Dense d20 scores **0.791 BPP and 0.215 CORE**; Remix d12 scores **0.814 and 0.172**. If that holds under measurement, at matched wall-clock RemixedLinear loses to a deeper dense model on both metrics. It is far better to scope the claim yourself than to have R4's "almost certainly loses" confirmed by an AC.

- [ ] 🔴 **Sweep chunk size N.** ⏱ 1 day at d4. **Nobody — not you, not any reviewer — has treated N as a variable.** N=64 appears to be arbitrary, and ablation 29A/29C shows chunk routing is not worse than per-token, so there is no evidence 64 is the ceiling. Sweep N ∈ {64, 128, 256, 512, 2048 (per-sequence)} reporting BPP *and* tok/s. The composition cost and the number of distinct W_eff matrices both fall linearly in N. If quality survives at N=512, your throughput story changes materially. This is the highest-upside cheap experiment available.

- [x] 🔴 **Re-examine the implementation before conceding the slowdown is intrinsic.** ⏱ 1 day. **Done, and the answer is mostly negative.** Shipped in `nanochat/gpt.py` as the Phase 31 flags: `--p31-chunk-route-impl grouped` (chunk-batched `torch._grouped_mm`, bit-exact against the compose path, tested in `tests/test_chunk_route_grouped.py`), `--p31-top1-gate switch`, `--p31-route-side {output,basis,narrow}`, `--p31-basis-side-templates`, `--p31-drop-basis-proj`, `--p31-template-delta-rank`. Layer-level speedup 1.12–1.19× at d8; end-to-end at d4 it was **+1.0%** (875k → 884k tok/s). A useful by-product: the grouped path exposed a zero-router-gradient bug that affected every top-1 configuration, now fixed. Two things remain worth saying in the paper: you implemented the batched-GEMM formulation R3 would assume you skipped, and the grouped fast path is currently *unreachable* for 29C because it requires `_qrouter is None`. Given Tier 0 shows the quantile router is already hard top-1, wiring it through would make the whole 29C configuration eligible. That is the highest-value remaining throughput work.

<details><summary>(superseded, kept for the record) the original bmm reformulation argument</summary>

§4.5 describes "weight composition via einsum, and a second einsum for the output." At d12 with batch 64, seq 2048, N=64 you have 32 chunks/seq × 64 = **2048 distinct W_eff matrices of 768×768 per projection**, i.e. ~2.4 GB of materialized weights written and read per projection per forward, × 72 projections. That is a memory-bandwidth catastrophe and it fully explains utilization dropping from 195 to 86 TFLOPS. Since W_eff is *constant within a chunk*, the natural formulation is: compose once per chunk, then a single **`torch.bmm`** of `[n_chunks, N, B] @ [n_chunks, B, d_out]` — cuBLAS-backed batched GEMM, no custom kernel. Your Triton attempt failed because it competed with cuBLAS; batched GEMM *uses* cuBLAS. Tile chunks so the W_eff working set fits L2. If this recovers even 2× you have a much better paper, and if it doesn't, you can say "we tried the batched-GEMM formulation" instead of only "we tried Triton" — which is what R3 will otherwise assume you didn't do.

</details>

- [ ] 🟠 **Full module-level throughput breakdown.** ⏱ 4h. R3 asked specifically: template gating, routing, output gating, memory movement, unfused ops. You have "attention ~5× slower, FFN ~8× slower" — that's not a breakdown, it's a symptom. Give per-op time and per-op bytes moved.

- [ ] 🔴 **Drop or fix the "distill to K=1" proposal.** ⏱ writing. R4 asks whether it retains the gain. As literally stated it cannot: K=1 scores 1.168 at d4 versus dense 1.170 — K=1 *is* a dense layer. Proposing it as the throughput fix is incoherent and R4 has spotted it. Either remove it, or restate what you actually mean (distilling a K=8 teacher into a dense student, which is a genuinely interesting *different* claim and should be flagged as untested future work, not as a fix in hand).

---

## TIER 2 — The ablations that are missing. ~2 days at d4.

**The under-appreciated hole in this paper:** the abstract names three key design choices — chunk-amortized routing, quantile balancing, identity-preserving init. Table 4 properly ablates **only the first**. Quantile balancing has *no ablation row anywhere*, and the gate is ablated only at K=1, where it contributes 0.002–0.005 BPP. Two of your three headline contributions currently rest on the 29-phase narrative rather than a controlled experiment at the operating point. R5 says a version of this; the AC says "more orthogonal ablations." Fix it.

- [ ] 🔴 **Quantile balancing ablation at K=8, d4.** ⏱ 6h. {quantile balancing / standard aux load-balancing loss / no balancing at all}. Report BPP *and* template utilization entropy. Right now the claim "eliminating the auxiliary losses required by standard MoE" has zero supporting evidence in the paper.
- [ ] 🔴 **Gate ablation at K=8, d4.** ⏱ 6h. {1+tanh centered gate / sigmoid gate / no output gate}, holding routing fixed. The gate is claimed as the innovation that "resolved the optimization friction" — prove it at the configuration you ship, not at K=1.
- [ ] 🔴 **Dense + intermediate LayerNorm baseline.** ⏱ 4h at d4, ~1 day at d8. Three reviewers and the metareview flag this. You already concede it. R4 is right that it "sits underneath every headline number, not just K=1." Note this experiment can only clarify or hurt — if dense+LN lands near 1.168, the factorization contributes nothing and *routing is the entire story* (which is a cleaner paper); if it lands at 1.15, your dense baseline is under-tuned and every gap shrinks. Not running it is worse than either outcome.
- [ ] 🟠 **Orthogonal sweep on the coupled knobs (R5).** ⏱ 1 day. Context dim d_c, gate rank r, basis size B — one-at-a-time from the canonical config at d4. Note the paper is already internally inconsistent here: §4.1 says gate rank **r = 16**, §3.3 says **r = 8**. Fix.
- [ ] 🟠 **Seeds.** ⏱ 1 day. 3 seeds × {dense, Remix} at d4. You cannot afford seeds at d12, but a d4 variance estimate lets you say "seed σ is X BPP; the d12 gap is Y×σ." That is a real argument. Right now you have none, and four reviewers plus the AC have noticed.

---

## TIER 3 — Analysis reviewers asked for. Cheap, mostly free. ~1 day.

- [x] 🔴 **Template specialization / utilization.** ⏱ 1 day. R1, R2, and R3 all asked; the AC lists it. Script: `scripts/paper_template_analysis.py`, which takes a trained checkpoint and reports, per projection and per layer: routing entropy and effective template count, usage histograms and load-balance CV, a **variance decomposition of α into within-sequence (position) and across-sequence (input) components**, the relative Frobenius deviation of `W_eff(chunk)` from its own mean, and the pairwise cosine between templates.

      python -m scripts.paper_template_analysis --ckpt <d12 checkpoint dir> --batches 20 --plot out/tmpl

  Two of those are the ones that matter. `|dW|/|W|` is the honest "how dynamic is this weight really" number, and mean pairwise template cosine tells you whether the bank collapsed: routing between templates that converged to the same matrix is decorative no matter how varied α looks. The script also prints a router-identity block, which on any 29-series checkpoint will report zero within-sequence variance and an effective template count of 1.0, for the reasons in Tier 0. **If the gain survives that, the story is "conditioning on the sequence", not "conditioning on position", and the paper should say so.**
- [ ] 🔴 **CORE per-task or category breakdown.** ⏱ 2h. R5's main ask; you already have the eval outputs. Look before you promise: at 300M params most CORE tasks sit near random, so check whether the aggregate gain is broad or concentrated in 2–3 tasks. Report "Remix > dense on k of 22 tasks" as a robustness statistic either way.
- [ ] 🔴 **Fix the differentiability sentence.** ⏱ 30min. R3 is straightforwardly correct: thresholding scores against θ_k is discrete, and "the resulting mask... yielding differentiable routing weights" is wrong as written. The accurate statement is that the *selection* is non-differentiable and receives no gradient (as in top-k MoE), while the *weights over selected templates* are differentiable through the masked softmax, and θ_k is a non-learned EMA buffer. Concede plainly; it is a writing error, not a method flaw.
- [ ] 🟠 **Define "matrix-shaped parameters."** ⏱ 10min. R3 asked. Say which tensors go to Muon vs AdamW, explicitly.
- [ ] 🟠 **Say what was tuned.** ⏱ 1h. R3's sharpest methodological point: 29 phases of search on RemixedLinear versus zero on the baselines is an unfair-comparison risk. State exactly what HP search each arm received. If the answer is "none for either, both use nanochat defaults," that is a *good* answer — say it.

---

## TIER 4 — Related work. No compute. Half a day. Do not skip.

R2 gives Quality 2 and Significance 2 largely on this, and the AC lists it. **The core operation W_eff = Σ α_k T_k is not new**, and defending its novelty will read badly.

- [ ] 🔴 **CondConv** (Yang et al. 2019) — same mathematical form (per-example mixing of parameter banks), applied to convolutions. Cite and differentiate.
- [ ] 🔴 **SMEAR** (Muqeeth, Liu & Raffel, TMLR 2024) — merges expert *parameters* by routing weights, explicitly to get differentiable routing. Very close to your mechanism *and* your motivation.
- [ ] 🔴 **Lory** (Zhong et al., COLM 2024) — the most dangerous omission. Fully differentiable expert *merging* for autoregressive LM pretraining with **causal segment routing**. That is chunk-amortized weight merging for exactly your application. Your delta is real but narrow (anchor on the current chunk's first token vs. the previous segment; per-projection rather than FFN-level) and you must state it precisely rather than in generalities.
- [ ] 🟠 Also add: Soft MoE (Puigcerver et al.), PEER / Mixture of a Million Experts (He, 2024), μMoE/multilinear MoE, DeepSeek fine-grained experts, LoRA-MoE variants, and Jacobs et al. (1991) for provenance.
- [ ] 🔴 **Rewrite the novelty claim.** ⏱ 2h. What survives honest scrutiny: (i) applying merged-parameter routing to **all six projections** rather than FFN-only; (ii) **quantile/EMA balancing** replacing the aux loss; (iii) the **centered 1+tanh identity-preserving gate**; (iv) the empirical finding that this **beats dense below 1B where MoE does not**. That is a real contribution list. "We invented template mixing" is not, and claiming it after Lory will cost you more than the citation gap already has.

---

## TIER 5 — Wanted but not feasible in the window. Say so plainly.

- [ ] 🟠 **d24 / ~1B active params** (R1, R4, R5, AC). **Revised: reachable, and the binding constraint is the token budget, not memory.** The AC's stated ceiling is "~290M active parameters", and d24 clears it: 5.32B total, **1.26B active** at chunk 256. Measured budget on a 141 GB H200 (all params, grads and Muon momentum are fp32; only embeddings are cast):

  | term | GB | scales with batch |
  |---|---|---|
  | params / grads / Muon momentum | 21.3 each | no |
  | bf16 bank copies saved for backward | 8.2 | no |
  | materialized `W_eff` | 8.2 × B | **yes** |
  | Muon `torch.stack` transient at step time | 18.1 | no |

  So ~72 GB is fixed cost that `--device-batch-size` cannot touch, which is why batch 1 still OOMed at chunk 64 (backward peak ~105 GB plus the 10–20 GB first-step spike). At chunk 256 the peak is ~80 GB and it fits; chunk 512 or `--p31-route-side narrow` buys more. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

  The real blocker: `--target-active-params 0` sets the horizon from *total* scaling params, so d24 asks for **49B tokens**, roughly a week of continuous H200 time and possibly more shards than you have. d20 is 28.5B tokens for 0.82B active. Options: run d24 on the active-param budget (6.3B tokens) and label the protocol change in one sentence, or accept d20 and state that 1B+ remains open. Recommend the former.
- [ ] 🟡 **d16 as a fourth point.** ⏱ ~1 week+. Probably infeasible, but if anything is running in the background, this is the one worth queuing.
- [ ] 🟠 **MoE at d8.** ⏱ 2–3 days. R4 asks; d12 is out of reach but **d8 may be feasible** and would materially strengthen "fine-grained beats coarse," which is currently supported at d4 only — precisely the scale where everyone already knows MoE is weak. Highest-value item in this tier.
- [ ] 🟠 **Second dataset at d4.** ⏱ 1 day. R4 asks whether the gain is nanochat/ClimbMix-specific. d4 is 37M params and 0.4B tokens — a FineWeb-Edu run is genuinely affordable and directly answers it. More feasible than reviewers assume; worth doing.

---

## If you have 7 days

Day 1: Tier 0 audits, including the router audit. Run `scripts/paper_throughput.py` and `scripts/paper_template_analysis.py`; both are written and both are hours, not days.
Days 2–3: chunk-size sweep, dense+LN baseline, and decide what the router section says.
Days 4–5: quantile ablation at K=8, gate ablation at K=8, d4 seeds (run in parallel).
Day 6: CORE breakdown, related-work rewrite, throughput and memory table.
Day 7: write. Stretch items if anything finished early: d4 second dataset, d8 MoE, d24 on the active-param budget.

## The one-line version

The FLOPs framing is not survivable and the AC says so in their first sentence; the active-parameter framing is. Retitle away from "compute-efficient", move the headline to matched active *parameters*, concede the wall-clock regression in the abstract rather than in Section 4.6, and find out whether dense d20 beats you at matched wall-clock before someone else does. Everything else on this list is cheaper than that decision and none of it substitutes for it.

# RemixedLinear — Rebuttal Task & Experiment Checklist

**Legend:** ⏱ wall-clock estimate · 🔴 decision-critical · 🟠 strongly helps · 🟡 nice-to-have
**Assumed window:** ~1 week, single-researcher heterogeneous compute, no institutional cluster.
**Reality check up front:** five reviews at Quality 3/2/2/3/2, one explicit borderline-reject, and a metareview that says the case for acceptance is "materially weakened." A rebuttal almost never moves that to accept. Optimize for (a) moving one or two reviewers, and (b) doing the work that makes the *next* submission strong — several items below will change what the paper claims, and it is much better to learn that now.

**Read the metareview as a spec.** The AC has effectively written the acceptance criteria: FLOPs-vs-wall-clock, single-seed, un-retrained dense baselines, d4-only MoE, LayerNorm confound, soft-MoE/SMEAR/CondConv positioning, differentiability, template-utilization analysis, orthogonal ablations, per-task CORE. Ten items. Six are cheap. Hitting those six *completely* is the realistic play.

---

## TIER 0 — Audits. Hours. Do before writing anything.

- [ ] 🔴 **Reconcile Table 3 vs Table 5 FLOPs.** ⏱ 1h. Table 3 reports d12 FLOPs as **7.6e8 for both** dense and Remix ("matched active FLOPs" — the paper's headline framing). Table 5 reports measured hardware FLOPs/token as **2.2e8 dense vs 3.6e8 Remix (1.64×)**. And §3.6 lines 125–126 state Remix uses "approximately 4d² active FLOPs per projection versus 2d² for dense, a **2× per-projection overhead**." These three cannot all be right. As written, **the paper's own complexity analysis contradicts its headline "matched active FLOPs" claim.** R4 got closest to this ("Hardware FLOPs are already 1.6× higher") but no reviewer stated it this sharply — a determined AC will. Work out exactly what each number counts (fwd only? fwd+bwd? projections only? attention included?) and state it in one place.
- [ ] 🔴 **"The improvement grows with scale" is false.** ⏱ 15min. Line 153 says: *"The improvement grows with scale: 7.5% at d4, 6.8% at d8, and 9.9% at d12."* That sequence decreases then increases. In absolute BPP the deltas are **0.088, 0.066, 0.089** — flat and non-monotonic. Nobody caught it. Delete the claim or restate it as "consistent across scales, with no evidence of growth."
- [ ] 🔴 **Which dense baselines are yours?** ⏱ 1h. Limitations says d12–d30 come from the nanochat leaderboard. So are d4 (1.170) and d8 (0.969) *your* runs? If yes, say so loudly — "our own matched-setup baselines at d4/d8, leaderboard at d12+" is a far better position than what R3/R4 currently believe. Also verify the leaderboard runs use the same dataset (ClimbMix), tokenizer, and 10.5× token ratio. If they don't, the d12 headline comparison is invalid and you need to know now.
- [ ] 🟠 **Sanity-check the exact tie.** ⏱ 15min. Remix d8 = 0.903 and Dense d12 = 0.903 to three decimals. Probably coincidence; confirm it isn't a copy error, because that pair carries the "2.25× fewer active params" claim.
- [ ] 🟠 **Audit Table 2 parameter accounting.** ⏱ 1h. Remix d12 total 792M vs dense 286M is 2.8×, not K=8×. Make sure the reader can reconstruct where the numbers come from; the FLOPs claims sit on top of them.
- [ ] 🟡 **CORE percentages.** ⏱ 15min. "36% above the dense power-law prediction" is 0.123 vs 0.090 — a 0.033 absolute difference on a centered-accuracy scale where most tasks are near random at 300M params. Report absolute deltas alongside percentages.

---

## TIER 1 — The wall-clock problem. This is the paper's fate. ~2 days.

Three reviewers and the metareview converge here. R3 states it most sharply and is right: *"If an algorithm decomposes a dense GEMM into hardware-costly operations, the slowdown is part of the method itself, not merely an implementation detail."*

- [ ] 🔴 **Measure throughput for the dense baselines at every depth — at random init.** ⏱ 3h. Throughput doesn't depend on trained weights, so you can measure dense d14/d20/d26/d30 tok/s on your H200 in an afternoon without any checkpoints. Combine with leaderboard BPP and CORE to plot **the quality-vs-wall-clock curve R4 explicitly asked for.** This is nearly free and it is the single most important number you don't have.

  **Be prepared for the answer.** From your own tables: dense d12 runs at 886k tok/s; dense d20 has ~3.8× the FLOPs, so ≈230k tok/s — essentially identical to Remix d12's 242k. Dense d20 scores **0.791 BPP and 0.215 CORE**; Remix d12 scores **0.814 and 0.172**. If that holds under measurement, **at matched wall-clock RemixedLinear loses to a deeper dense model on both metrics.** Verify it before you write anything. It is far better to report this yourself, scoped, than to have R4's "almost certainly loses" confirmed by an AC.

- [ ] 🔴 **Sweep chunk size N.** ⏱ 1 day at d4. **Nobody — not you, not any reviewer — has treated N as a variable.** N=64 appears to be arbitrary, and ablation 29A/29C shows chunk routing is not worse than per-token, so there is no evidence 64 is the ceiling. Sweep N ∈ {64, 128, 256, 512, 2048 (per-sequence)} reporting BPP *and* tok/s. The composition cost and the number of distinct W_eff matrices both fall linearly in N. If quality survives at N=512, your throughput story changes materially. This is the highest-upside cheap experiment available.

- [ ] 🔴 **Re-examine the implementation before conceding the slowdown is intrinsic.** ⏱ 1 day. §4.5 describes "weight composition via einsum, and a second einsum for the output." At d12 with batch 64, seq 2048, N=64 you have 32 chunks/seq × 64 = **2048 distinct W_eff matrices of 768×768 per projection**, i.e. ~2.4 GB of materialized weights written and read per projection per forward, × 72 projections. That is a memory-bandwidth catastrophe and it fully explains utilization dropping from 195 to 86 TFLOPS. Since W_eff is *constant within a chunk*, the natural formulation is: compose once per chunk, then a single **`torch.bmm`** of `[n_chunks, N, B] @ [n_chunks, B, d_out]` — cuBLAS-backed batched GEMM, no custom kernel. Your Triton attempt failed because it competed with cuBLAS; batched GEMM *uses* cuBLAS. Tile chunks so the W_eff working set fits L2. If this recovers even 2× you have a much better paper, and if it doesn't, you can say "we tried the batched-GEMM formulation" instead of only "we tried Triton" — which is what R3 will otherwise assume you didn't do.

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

- [ ] 🔴 **Template specialization / utilization.** ⏱ 1 day. R1, R2, and R3 all asked; the AC lists it. From an existing d12 checkpoint: routing-weight entropy per projection, template usage histograms, whether Q/K/V/O/FFN projections specialize differently, how routing correlates with token type or position, and how much α actually varies across chunks. **If α turns out to be nearly uniform, you need to know that** — it would mean the gain comes from the extra parameters and the LayerNorm, not from routing, and it would explain why chunk-64 ≈ per-token.
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

- [ ] ❌ **d20 / ~1B active params** (R1, R4, R5, AC). ~9B tokens at Chinchilla. Weeks on your hardware. Not happening. State the compute honestly and commit to it as the next milestone.
- [ ] 🟡 **d16 as a fourth point.** ⏱ ~1 week+. Probably infeasible, but if anything is running in the background, this is the one worth queuing.
- [ ] 🟠 **MoE at d8.** ⏱ 2–3 days. R4 asks; d12 is out of reach but **d8 may be feasible** and would materially strengthen "fine-grained beats coarse," which is currently supported at d4 only — precisely the scale where everyone already knows MoE is weak. Highest-value item in this tier.
- [ ] 🟠 **Second dataset at d4.** ⏱ 1 day. R4 asks whether the gain is nanochat/ClimbMix-specific. d4 is 37M params and 0.4B tokens — a FineWeb-Edu run is genuinely affordable and directly answers it. More feasible than reviewers assume; worth doing.

---

## If you have 7 days

Days 1: Tier 0 audits + dense-throughput measurements + the wall-clock curve.
Days 2–3: chunk-size sweep, bmm reformulation + re-profile, dense+LN baseline.
Days 4–5: quantile ablation at K=8, gate ablation at K=8, d4 seeds (run in parallel).
Day 6: template specialization analysis, CORE breakdown, related-work rewrite.
Day 7: write. Stretch items if anything finished early: d4 second dataset, d8 MoE.

## The one-line version

The FLOPs framing is not survivable; the active-parameter framing is. Find out today whether dense d20 beats you at matched wall-clock, and rebuild the claim around what's left.

# RemixedLinear — Assessment and Rebuttal Draft

---

## Part 1 — What isn't working

### The headline claim is not the claim your evidence supports

The paper is titled and framed around **compute efficiency**. Your own Table 5 says hardware FLOPs are 1.6× higher and throughput is 3.7× lower. Those are not compatible, and four of five reviewers plus the AC noticed.

It's worse than a framing problem. Three numbers in the paper contradict each other:

- **Table 3:** d12 FLOPs = 7.6e8 for *both* dense and Remix — "matched active FLOPs," the basis of every headline comparison.
- **Table 5:** measured hardware FLOPs/token = 2.2e8 dense vs **3.6e8** Remix.
- **§3.6, lines 125–126:** *"RemixedLinear uses approximately 4d² active FLOPs per projection versus 2d² for dense, a 2× per-projection overhead."*

Your own complexity analysis says the method costs 2× per projection, and your own measurement says 1.64× overall. So "matched active FLOPs" appears to be an accounting convention that counts the *weighted-sum* operator once while the implementation pays for composing it. R4 got closest ("Hardware FLOPs are already 1.6× higher") but nobody stated it this directly. An AC checking Table 3 against §3.6 will.

### Work out the wall-clock comparison before you write anything

R4 says at equal wall-clock the method "almost certainly loses to a deeper dense model." That is computable from your own tables, and the answer looks bad:

| | Throughput | BPP | CORE |
|---|---|---|---|
| Remix d12 | 242k tok/s | 0.814 | 0.172 |
| Dense d12 | 886k tok/s | 0.903 | 0.114 |
| Dense d20 (est. ~230k tok/s) | ~same as Remix d12 | **0.791** | **0.215** |

Dense d20 has ~3.8× the FLOPs of d12, so roughly 886/3.8 ≈ 230k tok/s — essentially identical to Remix d12. And it wins on both metrics. **Measure it before you send anything.** Dense throughput doesn't require trained weights: instantiate d14/d20/d26/d30 at random init and time them. It's an afternoon.

If it holds, the "compute-efficient" framing is dead and defending it will cost you more than conceding it. What survives, and it's genuinely worth something:

- **At matched active parameters:** Remix d8 (127M) = dense d12 (286M) at 0.903 BPP. Real, clean, 2.25×.
- **At matched *measured* FLOPs:** Remix d12 at 3.6e8 vs dense at ~3.6e8 ≈ d14 (0.869). Remix's 0.814 still wins by ~0.055.
- **At matched wall-clock:** you lose.

Say exactly that, in the abstract, in that order. A scoped claim you defend precisely reads as rigor. An unscoped claim that a reviewer punctures reads as overselling — which is R4's word for it.

### Two of your three headline contributions have no controlled ablation

The abstract names chunk-amortized routing, quantile balancing, and identity-preserving init as "the key design choices that enable dynamic linear layers to outperform static baselines."

Table 4 ablates **only the first**. There is **no quantile-balancing row anywhere in the paper** — not against an aux loss, not against no balancing. The gate is ablated only at K=1, where it contributes 0.002–0.005 BPP, and never at the K=8 configuration you actually ship. So two of three contributions rest on the 29-phase narrative rather than on a controlled experiment at the operating point. R5 says a version of this and the AC asks for "more orthogonal ablations"; neither states it this starkly. Both experiments are d4 runs. Do them.

### The novelty position is weak and defending it will backfire

W_eff = Σ α_k T_k is **CondConv** (2019) for convolutions and **SMEAR** (TMLR 2024) for experts. Chunk-amortized weight merging with causal segmentation for autoregressive LM pretraining is **Lory** (COLM 2024). R2 cited all three and scored Originality 3 anyway — that's generous. Arguing the core mechanism is novel invites someone to read Lory carefully.

What is actually yours: all six projections rather than FFN-only; quantile/EMA balancing instead of an aux loss; the centered 1+tanh gate; and the empirical result that this beats dense below 1B where MoE doesn't. That's a legitimate contribution list. Lead with it and cite the rest generously.

### Smaller things that are just wrong

- **"The improvement grows with scale: 7.5% at d4, 6.8% at d8, and 9.9% at d12"** (line 153). That sequence goes down then up. In absolute BPP: 0.088, 0.066, 0.089 — flat, non-monotonic. No reviewer caught it. Delete or restate.
- **R3's differentiability objection is correct.** Thresholding scores against θ_k is discrete. "The resulting mask is applied before a masked softmax, yielding differentiable routing weights" is wrong as written. Concede.
- **"Distill to K=1 at inference to recover dense throughput"** (§4.5) is incoherent as stated — K=1 scores 1.168 vs dense 1.170, i.e. K=1 *is* dense. R4 asked directly whether it retains the gain. It cannot. Remove it or restate it as distillation into a dense student, which is a different and untested claim.
- **r = 8 in §3.3, r = 16 in §4.1.**

### What's genuinely in your favour

R4 — your harshest scorer on significance — calls the paper "unusually honest," singles out the 29-phase appendix as real engineering knowledge, and calls quantile balancing "a genuinely nice simplification." The AC says "promising and unusually transparent." That credit is worth protecting: it is the reason a scoped, self-critical rebuttal will land better here than a defensive one. Do not spend it defending the FLOPs framing.

### Strong vs. answerable

**Strong and hard to counter:** wall-clock regression (R3, R4, AC); single-seed with no variance (R2, R4, AC); scale ceiling at 290M (R1, R4, R5, AC); leaderboard baselines not retrained (R3, R4, AC); missing soft-MoE/SMEAR/CondConv/Lory (R2, AC).

**Legitimate but cheaply fixable:** LayerNorm confound; differentiability wording; per-task CORE; template specialization; orthogonal ablations; "matrix-shaped parameters"; what was tuned on which arm.

**Overreach, gently:** R3's *"the experiments feel rushed"* — 29 phases plus three scales plus CORE plus throughput plus a failed-kernel report is not rushed; it's under-resourced, which is different and stated. And R3's suspicion that dense baselines got no tuning cuts both ways: if neither arm was tuned beyond nanochat defaults, that's a *fair* comparison, and you should say so rather than let the suspicion stand.

**Not a defect, just a limit:** R1's "would it transfer to instruction tuning / reasoning." Out of scope for a pretraining architecture paper; acknowledge and move on.

---

## Part 2 — Rebuttal draft

> Fill every `[BRACKET]` with a measured number. Do not send a bracket, and do not promise an experiment you aren't running. NeurIPS response boxes are tight — trim the per-reviewer replies before the global section.

### Global response

We thank all five reviewers and the AC. The reviews converge on one issue and we think they are right about it: **we framed the contribution as compute efficiency when our evidence supports efficiency in active parameters and analytical FLOPs, but not in wall-clock time.** Rather than defend that framing, we are correcting it and reporting the measurements that make the scope precise.

**1. The wall-clock comparison, measured.** We have measured inference throughput for dense baselines at every depth on the same H200 (batch 64, seq 2048), which lets us plot quality against wall-clock alongside quality against FLOPs, as R4 asked. Results:

[TABLE: depth / tok/s / BPP / CORE for dense d12–d30 and Remix d8, d12]

The honest summary is a three-line hierarchy, which we now state in the abstract:
- At matched **active parameters**, RemixedLinear d8 (127M) matches dense d12 (286M) at 0.903 BPP — a 2.25× reduction.
- At matched **measured hardware FLOPs**, Remix d12 (3.6e8/token) is best compared against dense ≈d14, and improves BPP by [X].
- At matched **wall-clock throughput**, Remix d12 (242k tok/s) is comparable to dense ≈d[20], which achieves [BPP] / [CORE]. **In this regime the current implementation does not win**, and we say so in the abstract and introduction rather than only in §4.5.

**2. We are correcting an inconsistency reviewers were right to probe.** Table 3 reported analytical active FLOPs as matched, while §3.6 states a 2× per-projection overhead and Table 5 measures a 1.64× hardware-FLOPs gap. These count different things and we conflated them. The revision reports a single FLOPs convention throughout, states measured hardware FLOPs in the main results table, and no longer describes the comparison as "matched active FLOPs" without qualification.

**3. New ablations at the operating configuration.** We agree with R5 and the AC that our ablations were not orthogonal, and on re-reading we found that two of our three claimed design choices were not controlled at K=8 at all. We have run at d4: quantile balancing vs. auxiliary load-balancing loss vs. no balancing [results]; the centered 1+tanh gate vs. sigmoid vs. no gate, at K=8 [results]; and one-at-a-time sweeps of context dimension, gate rank, and basis size [results].

**4. The LayerNorm confound is now resolved, not deferred.** Dense + intermediate LayerNorm at d4 gives [X] BPP versus 1.170 dense and 1.168 for K=1 RemixedLinear. [Interpretation follows the number honestly.]

**5. Variance.** We ran 3 seeds of both dense and RemixedLinear at d4: σ = [X] BPP. We cannot afford seeds at d8/d12 and do not claim otherwise; we report the d4 σ so readers can judge the d8/d12 gaps ([0.066] and [0.089]) against it.

**6. Related work.** We have substantially rewritten §2 to position against CondConv, SMEAR, Lory, Soft MoE, and PEER. We say more in our reply to R2 — briefly, we agree the *mechanism* of mixing parameter banks by routing weights is prior art, and we have removed any implication otherwise.

**7. Corrections.** We have removed the claim that the improvement grows with scale (the absolute deltas — 0.088, 0.066, 0.089 — do not support it); corrected the description of differentiability in quantile routing (R3 is right); removed the "distill to K=1" proposal (R4 is right that it cannot retain the gain, since K=1 is a dense layer); and fixed the gate-rank inconsistency between §3.3 and §4.1.

**What we could not do.** We could not train at d20 or ~1B active parameters within the response window; on our compute that is weeks, not days, and we state it as the required next milestone rather than promising it. [If d8 MoE / second dataset landed, name them here; otherwise say they did not.]

---

### To Reviewer 1

Thank you for the careful and generous reading.

**On scale (Q1).** We cannot resolve this empirically here — d20 is weeks of training on our hardware — and we would rather say that than offer speculation dressed as evidence. What we can offer: our three points give absolute BPP gaps of 0.088 / 0.066 / 0.089 over dense at d4/d8/d12. We had previously described this as growing with scale; it does not, and we have corrected that sentence. The honest reading is that the gain is *stable* over the range tested, with no evidence about behaviour beyond ~290M active parameters — which, as R4 notes, is precisely where MoE-style methods usually change character. We have moved this from a soft limitation to an explicit threat to the scaling conclusion.

**On mechanism (Q2).** We agree this was the weakest part of our analysis and we have added it. From a trained d12 checkpoint we now report: routing-weight entropy per projection type (Q/K/V/O/FFN), template utilization histograms, and how routing varies across chunks and positions [results]. [Report what you find, including if routing turns out to be closer to uniform than expected — that finding is publishable and hiding it is not.] We have also added the controlled ablations that were missing: the centered gate and quantile balancing were previously supported only by our design-history narrative and by a K=1 ablation, not by a controlled experiment at K=8. Those experiments now exist [results].

**On positioning.** We agree, and R2 identified specific omissions (CondConv, SMEAR, Lory) that we should have engaged with. §2 is substantially rewritten; see our reply to R2.

**On downstream transfer.** Instruction tuning and reasoning evaluation are outside what we can support at ≤290M parameters, and we have scoped the claims to pretraining quality and in-context CORE performance accordingly.

---

### To Reviewer 2

Thank you — both criticisms are correct and we have acted on both.

**On related work.** This is a fair hit and we do not contest it. The operation W_eff = Σ α_k(c)·T_k is not new: CondConv [3] introduced per-example mixing of a parameter bank for convolutions, SMEAR [1] merges expert parameters by routing weights specifically to obtain differentiable routing, and Lory [2] applies fully differentiable expert merging to autoregressive LM pretraining with causal segment routing. We should have cited all three and we have added them, along with Soft MoE, PEER, and Jacobs et al. (1991).

**On positioning relative to CondConv and SMEAR (Q1).** Precisely:

- *CondConv* mixes a bank of convolution kernels with per-**example** routing coefficients. RemixedLinear applies the same mixing form to linear projections with per-**chunk**, causally-anchored routing inside an autoregressive sequence model, where per-example routing is not available.
- *SMEAR* merges expert parameters to avoid non-differentiable discrete routing, at the granularity of whole expert modules. RemixedLinear operates at the granularity of individual projections (six per block), giving 6K routing decisions per block rather than one.
- *Lory* is the closest prior work and shares our motivation directly. The differences are narrower than we would like: Lory routes using the *previous* segment's representation to preserve causality; we anchor on the *first token of the current* chunk. Lory operates at FFN-expert granularity; we operate per-projection with a shared basis and an explicit output gate. We do not claim the merging mechanism as our contribution.

What we do claim, and would ask the reviewer to weigh: (i) applying merged-parameter routing to all six projections including Q/K/V/O rather than FFN only; (ii) replacing the auxiliary load-balancing loss with an EMA-quantile rule, which removes a loss term and its hyperparameter; (iii) the centered 1+tanh gate that makes the layer exactly output-equivalent to dense at initialization; and (iv) the empirical result that this configuration beats dense **below 1B active parameters**, the regime where MoE reliably does not. We have rewritten the abstract and §2 so the contribution reads as this list rather than as the mixing mechanism.

**On "compute-efficient" in the title (Q2).** You are right that this is in tension with Table 5, and we should not have used the term without qualification. See the global response for the measured wall-clock curve and the three-line claim hierarchy. We are retitling and rewriting the abstract so that the efficiency claim is stated in active parameters and measured FLOPs, with the wall-clock regression stated in the same breath rather than deferred to §4.5.

**On template specialization (Q3).** Added; see our reply to R1 and the global response. [Results.]

**On error bars.** Added at d4 (3 seeds, both arms, σ = [X]); infeasible at d8/d12 and we say so rather than implying otherwise.

---

### To Reviewer 3

Thank you — several of these are corrections we should have caught ourselves.

**On differentiability (Q1).** You are right and our sentence was wrong. The accurate description: the per-template EMA threshold θ_k is a non-learned buffer updated from batch quantiles and receives no gradient; the selection (score > θ_k, unioned with a top-k fallback) is a discrete operation and is **not** differentiable; gradients flow only to the logits of *selected* templates through the masked softmax, exactly as in standard top-k MoE. What the quantile rule buys is not differentiability — it is balance without an auxiliary loss. We have rewritten §3.2 accordingly and thank you for catching it.

**On throughput (main concern).** We accept this fully, including the framing: if a method decomposes a dense GEMM into hardware-costly operations, the slowdown belongs to the method. We are no longer treating it as an implementation detail. Two responses:

First, the honest comparison. We have measured dense throughput at every depth and now report quality vs. wall-clock alongside quality vs. FLOPs [table]. At matched throughput, RemixedLinear d12 is comparable to dense d[20], which achieves [BPP]/[CORE]. We state this in the abstract.

Second, we investigated the source rather than only reporting the symptom. Since W_eff is constant within a chunk, at d12/batch 64/N=64 the implementation materializes ~2048 distinct 768×768 effective matrices per projection, which is a memory-bandwidth bound rather than a compute bound — consistent with utilization falling to 86 TFLOPS. We have [reformulated the computation as a per-chunk `torch.bmm` batched GEMM, avoiding custom kernels entirely: throughput improves from 242k to [X] tok/s]. We have also swept chunk size N ∈ {64…2048}, which we had not previously treated as a variable: [results]. [Report honestly if neither closes the gap — "we tried the batched-GEMM formulation and it did not help, for reason X" is a much stronger statement than the Triton anecdote alone.]

**On the throughput breakdown (Q2).** Added: per-operation time and bytes moved for routing, template composition, the mixing GEMM, the output gate, and the context stream [table].

**On what was tuned (methodological fairness).** This is the point we most want to correct, because the current text invites the wrong inference. [Choose the true version: *Neither arm received hyperparameter search: both RemixedLinear and the dense/MoE baselines use nanochat defaults for optimizer, LR schedule, and batch size, and the 29 phases were architecture search — which components to include — not hyperparameter tuning of the final configuration.* / Or state exactly what was tuned on each arm.] We have added an explicit paragraph on this.

**On "matrix-shaped parameters."** Poor wording on our part. Muon is applied to 2-D weight tensors ([enumerate: attention and FFN projections, basis projections, and template matrices]); AdamW is applied to embeddings, norms, biases, and scalars ([enumerate]). Now stated explicitly.

**On Section 3 clarity.** Fair. We have restructured §3 to introduce one concept per subsection in dependency order (basis projection → template bank → routing → chunk amortization → gate → context stream) with a running dimension example, and moved the FLOPs accounting adjacent to Figure 1.

**On "the experiments feel rushed."** We would gently push back on the word while accepting the substance. The work is under-resourced rather than hurried — no institutional compute, heterogeneous GPUs, checkpoint resumption — and we agree that the *consequences* (three points, single seeds, d4-only MoE, borrowed baselines) are real limitations regardless of cause. We have foregrounded the compute situation in §4.1 instead of in an appendix paragraph, so readers can weight the evidence appropriately from the start.

---

### To Reviewer 4

Thank you for an unusually precise review; several of your questions identified things we had not worked out ourselves.

**On the wall-clock framing.** You are right and your prediction is testable from our own numbers, so we tested it. [Report the measured curve.] At matched throughput, Remix d12 is comparable to dense d[20] at [BPP]/[CORE]. We now state the three-regime hierarchy — active parameters, measured FLOPs, wall-clock — in the abstract, with the wall-clock result included rather than deferred. We agree the previous framing oversold deployability.

**On the K=1 distillation proposal (Q2).** You are right that this is central, and on reflection it is worse than untested: as written it cannot work, because K=1 is a dense layer (1.168 vs. 1.170 at d4). We have removed the proposal. What we should have written — and now flag as untested future work rather than as a fix in hand — is distillation of a K=8 *teacher* into a dense *student*, which would be a different claim about RemixedLinear as a training-time device.

**On seeds (Q3).** 3 seeds at d4 for both arms, σ = [X] BPP; d8/d12 remain single-run and we cannot change that. We now report the d4 σ next to the d8/d12 gaps (0.066 and 0.089) and treat single-seed-plus-leaderboard-baselines as an explicit threat to the scaling conclusion, as you suggest, rather than as a limitation footnote.

**On dense + intermediate LayerNorm (Q4).** Run: [X] BPP at d4. You are right that this sits underneath every headline number, not only K=1. [If it explains the K=1 gap: *This confirms the K=1 factorization contributes nothing on its own and that routing is the entire source of the gain — which we think is a cleaner claim than the one we made.* / If it doesn't: *report it.*]

**On a second dataset or codebase (Q5).** [If run: results at d4 on [dataset].] [If not: *We could not complete this in the window. We agree it is the right test for framework-specific tuning and flag it as an open threat.*]

**On MoE at d8/d12 (Q6).** [If d8 landed: results.] [If not: *d12 MoE is out of reach on our compute. We agree that a d4-only MoE comparison is weak evidence for "fine-grained beats coarse," precisely because d4 is the regime where MoE is already known to underperform, and we have downgraded that claim accordingly.*]

**On the formatting notes.** All accepted: the abstract now reconciles the efficiency claim with Table 5 in place; we have added a phase → hypothesis → result → lesson summary table for the 29 phases and surfaced the FM1–FM5 taxonomy in the main text; Figures 2 and 3 are enlarged with fit parameters printed; and Table 6 now names the dense depth each Remix point matches on CORE.

---

### To Reviewer 5

Thank you — the ablation-design critique is the one we found most useful, and acting on it revealed a gap we had not noticed.

**On per-task CORE (W1).** Added: category-level averages across the five ability groups plus full per-task scores in the appendix [results]. We also report the number of tasks on which RemixedLinear exceeds dense at matched active FLOPs ([k]/22), since at ≤290M parameters many individual CORE tasks sit near random and an aggregate can be driven by a small subset. [If the gain is concentrated, say so — it is a more informative result than a clean aggregate.]

**On scale (W2).** We agree three points are insufficient and we cannot add a fourth within the window; d20 is weeks on our hardware. We have removed the language implying an established scaling law, corrected the claim that the improvement grows with scale (the absolute deltas are 0.088 / 0.066 / 0.089, which is flat), and now describe the result as a consistent offset over a narrow range rather than a shifted scaling curve.

**On coupled design choices (W3).** This is where your review changed the paper. Re-examining Table 4 against your point about K=8 versus K=1, we found that **two of our three claimed key design choices were never ablated at the shipped configuration**: quantile balancing has no ablation row at all, and the gate was isolated only at K=1, where it contributes 0.002–0.005 BPP. Those experiments now exist at d4:

- Balancing at K=8: quantile [X] / auxiliary loss [Y] / none [Z] BPP, with template utilization entropy [values].
- Gate at K=8: centered 1+tanh [X] / sigmoid [Y] / none [Z] BPP.
- One-at-a-time: context dimension [results], gate rank [results], basis size [results], intermediate LayerNorm [results].

We have restructured Table 4 into the aligned form you proposed — same routing × different gates, same gate × different routing — with the MoE baselines separated out. We also fixed an inconsistency you may have noticed: gate rank is given as 8 in §3.3 and 16 in §4.1; the correct value is [X].

---

## Part 3 — Tone and tactics

- **Lead with the concession, not the defence.** Four reviewers and the AC hit the same thing. A response that opens by conceding the framing and immediately produces the measured wall-clock curve reads as rigor; twenty individual rebuttals of the same objection read as resistance.
- **Report the unflattering number yourself.** If dense d20 beats you at matched throughput, saying so — with the active-parameter claim intact beside it — is the strongest available move. R4 already predicted the result; confirming it voluntarily costs you far less than being confirmed by an AC.
- **Don't defend novelty of the mixing mechanism.** Concede CondConv/SMEAR/Lory fully and fight on the contribution list instead. R2 gave you Originality 3 despite citing all three; that goodwill evaporates if you argue.
- **Protect the credit you have.** R4 and the AC both praised the paper's honesty. That is the one asset a rebuttal can spend or compound. Compound it.
- **Never promise an experiment you aren't running.** With a metareview already written, unfulfilled promises are strictly negative.
- **Answer every direct question in the first sentence of its reply.** R3's Q1 and R4's Q2 and Q4 are yes/no-shaped. Answer, then explain.
- **Be realistic about the outcome.** Five reviews at Quality 3/2/2/3/2 with a negative metareview very rarely become an accept. Write the rebuttal to move a reviewer and, more importantly, to surface what the next version of this paper has to claim. The chunk-size sweep, the batched-GEMM reformulation, and the wall-clock curve are worth doing whether or not this submission survives — they determine whether there is a paper here at all.

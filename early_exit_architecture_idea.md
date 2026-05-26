This is a much more interesting architecture than standard MoD. Let me work through all three parts.

---

## Part 1 — The "Exit to Final Layer" Architecture

Your proposal is conceptually clean: tokens exit early but every token participates in the final layer. So the forward pass looks like:

```
All tokens: Layer 0 → Layer 1 → ... 
                ↓ (some exit at layer 2)
                ↓ (some exit at layer 5)  
                ↓ (some exit at layer 9)
                ↓ (remaining run all layers)
                        ↓
            ALL tokens → Final Layer N
                        ↓
                   Output / Loss
```

This is clever for several reasons.

**It solves the attention mask problem partially.** Tokens that exit early don't need to maintain fresh representations through middle layers. Their hidden state is frozen at their exit layer and carried to layer N. Tokens still running in middle layers only attend to each other — and there are fewer of them, so attention is cheaper. The final layer sees everyone again and can integrate information across all exit depths.

**It gives the final layer a meaningful job.** The final layer becomes an integration layer — it takes representations from different depths and produces a coherent output. This is actually well-motivated by your TRS findings: the 90%+ never-stable tokens need the final layer to complete their processing, while the stable tokens just need their early representation projected correctly.

**The attention mask problem in middle layers is real but bounded.** At layer k, tokens that exited before k are absent from the attention computation. Tokens still present at layer k can only attend to each other. This is the key technical challenge — a token at layer 7 can't attend to a token that exited at layer 3. You need to decide what to do with those absent tokens in the attention matrix.

The two options are:

**Option A — Masked attention:** Tokens present at layer k attend only to other tokens present at layer k. Absent tokens are masked out. Simple but means later tokens lose access to easy tokens' representations mid-network.

**Option B — Frozen KV injection:** When a token exits at layer k, its key and value vectors are frozen and injected into all subsequent layers' attention computation. The token doesn't run the FFN or update its hidden state, but its frozen KV remains available. This preserves attention connectivity at the cost of storing frozen KVs for exited tokens.

Option B is significantly better for model quality and is the one worth pursuing. The memory cost is bounded — you're storing O(n_exited × d_model) KV pairs per layer, which is manageable.

---

## Part 2 — Using POS and Frequency as Routing Signal

Your TRS findings give you three concrete priors you can inject into the router.

**Frequency prior.** You found ρ = -0.32 to -0.45 between log frequency and critical depth across models. This means you can build a frequency-based initial routing bias directly. Before training starts, compute log frequency for every token in the vocabulary from the training corpus. Normalize to [0, 1]. Use this as an initial bias toward early exit for high-frequency tokens.

Concretely, the router at layer k produces a score for each token:

```
exit_score(token_i, layer_k) = router_network(h_i^k) + α × freq_bias(token_i)
```

Where `freq_bias` is the normalized log frequency and α is a learnable or fixed scalar. High-frequency tokens start with a bias toward exiting early. The router_network learns to refine this during training.

**POS prior.** Your findings show a consistent ordering: punctuation exits earliest, then function words, then content words, then numbers. You can encode this as a categorical prior. Assign each vocabulary token a POS category at initialization (using spaCy on the vocabulary, same as your diagnostic experiment). Map categories to initial exit layer targets:

```python
POS_EXIT_PRIOR = {
    'punctuation': 0.15,  # bias toward very early exit
    'function':    0.35,  # bias toward early-mid exit  
    'subword':     0.25,  # bias toward early exit
    'content':     0.65,  # bias toward mid-late
    'number':      0.75,  # bias toward late
    'other':       0.50,  # neutral
}
```

These are prior probabilities of exiting at each layer, not hard assignments. The router learns to deviate from them as training progresses.
This is how we get POS category:

```python
# ---- POS tagging with spaCy ----
# We tag unique token strings rather than every occurrence

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Define coarse POS categories
POS_MAP = {
    'NOUN': 'content',  'PROPN': 'content', 'VERB': 'content',
    'ADJ':  'content',  'ADV':  'content',
    'DET':  'function', 'ADP':  'function', 'CONJ': 'function',
    'CCONJ':'function', 'SCONJ':'function', 'PRON': 'function',
    'AUX':  'function', 'PART': 'function',
    'PUNCT':'punctuation', 'NUM': 'number',
    'X':    'other',    'SYM':  'other',    'INTJ':'other',
}

import re

def is_genuine_subword_fragment(raw_token: str, clean: str) -> bool:
    """
    Returns True only if this is a genuine subword continuation fragment.
    
    Genuine fragments look like:
      - Pure suffixes: 'ing', 'ed', 'ers', 'ify', 'tion', 'ment'
      - Pure prefixes: 'aut', 'odial', 'un', 'pre'
      - Mid-word chunks: 'ooks', 'imp', 'rid', 'uid', 'ug'
    
    NOT fragments:
      - Capitalised words: 'Facebook', 'Background', 'The', 'Starting'
      - Known complete lowercase words: 'allowed', 'get', 'for', 'based'
      - Tokens with hyphens that look like compound parts: '-for', '-based'
      - Acronyms: 'TCP', 'CC', 'AT', 'DS', 'A', 'F', 'B'
    """
    # No word boundary marker already confirmed by caller
    
    # Reject if capitalised — capitalised tokens after punctuation/newline
    # are word starts even without the marker
    if clean and clean[0].isupper():
        return False
    
    # Reject hyphen-prefixed tokens — these are compound word parts
    # that start new morphological units e.g. '-based', '-for'
    if raw_token.startswith('-'):
        return False
    
    # Reject dot-prefixed tokens e.g. '.S'
    if raw_token.startswith('.'):
        return False
    
    # Reject apostrophe tokens e.g. "'s" — these are clitics, not subwords
    if raw_token.startswith("'"):
        return False
    
    # Reject pure uppercase — acronyms
    if clean.isupper() and len(clean) <= 4:
        return False
    
    # Reject tokens that are common complete words
    # Use a small high-coverage stopword set
    COMPLETE_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'can', 'get',
        'got', 'let', 'set', 'put', 'run', 'for', 'and', 'but', 'or',
        'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'only',
        'own', 'same', 'than', 'too', 'very', 'just', 'allowed', 'based',
        'action', 'phone', 'current', 'starting', 'background',
        'n', 's', 'x', 'y', 'z',  # single letter ambiguous ones
    }
    if clean.lower() in COMPLETE_WORDS:
        return False
    
    # What remains should be genuine fragments:
    # lowercase, no special prefix, not a complete word, has alpha chars
    return bool(re.match(r'^[a-z][a-z]*$', clean)) and len(clean) >= 2


def get_pos_category(token_id: int, token_str: str, tokenizer) -> str:
    
    raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    clean     = token_str.strip()
    
    has_word_boundary = (
        raw_token.startswith('▁') or
        raw_token.startswith('Ġ') or
        raw_token.startswith('<') or
        raw_token.startswith('\n')
    )
    
    has_alpha = any(c.isalpha() for c in clean)
    
    # Only attempt subword classification if no boundary marker
    if not has_word_boundary and has_alpha:
        if is_genuine_subword_fragment(raw_token, clean):
            return 'subword'
        # Falls through to normal POS tagging if not a genuine fragment
    
    if not clean or not clean.isascii():
        return 'other'
    if clean.replace('.','').replace(',','').isdigit():
        return 'number'
    if all(not c.isalnum() for c in clean):
        return 'punctuation'
    
    doc = nlp(clean)
    if doc:
        return POS_MAP.get(doc[0].pos_, 'other')
    return 'other'
```

**Domain prior.** Your findings show code requires deeper processing than natural language in trained models. During training you know which domain each batch comes from. You can condition the router on a domain embedding:

```
exit_score = router_network(h_i^k, domain_embed) + α × freq_bias + β × pos_bias
```

This lets the router learn domain-specific exit patterns rather than one global routing policy.

The key insight is that these priors don't constrain the router — they initialize and regularize it. The router is free to learn anything during training. But starting from an informed prior means it doesn't waste the first portion of training discovering things you already know from the diagnostic.

---

## Part 3 — The Exploration/Warmup Loss

This is the most novel part of what you're proposing and I think it's genuinely interesting. Let me develop it properly.

The core problem you're identifying is: how does the model discover which tokens can safely exit early without knowing in advance what "safely" means for an untrained model? You need some form of exploration before committing to a routing policy.

**Phase 1 — Dense Warmup (steps 0 to W)**

Train normally with all tokens running all layers. No routing, no early exit. The goal is to let representations stabilize enough that the router has something meaningful to route on. Based on your Pythia training evolution findings, the frequency-depth correlation emerges strongly by step 128-2000, so W doesn't need to be large — maybe 1-2% of total training steps.

During this phase, run the router network in observation mode — compute exit scores but don't act on them. This gives you a distribution of exit scores across tokens and layers that you can use to calibrate the routing threshold.

**Phase 2 — Exploration Phase (steps W to W+E)**

Introduce a specific loss to discover which tokens benefit from early exit. The loss has two components:

*Reconstruction loss:* For a token that exits at layer k, measure how well a lightweight translator (same as your TunedLens translator) can predict the final layer's representation from the exit-layer hidden state. Formally:

```
L_reconstruct(i, k) = KL(p_final(token_i) || p_translator(h_i^k))
```

Low reconstruction loss at layer k means the token's representation at k is already close to what the full model would produce. This is exactly your TRS signal, but now computed during training rather than post-hoc.

*Efficiency loss:* A differentiable penalty that rewards earlier exits:

```
L_efficiency = (1/N) Σ_i (exit_layer_i / N_layers)
```

This encourages the router to exit tokens as early as possible subject to the reconstruction loss being low enough.

The combined exploration loss is:

```
L_explore = L_LM + λ_r × L_reconstruct + λ_e × L_efficiency
```

Where L_LM is the standard language modeling loss. During exploration, λ_r is high (reconstruction quality matters a lot) and λ_e is low (efficiency is secondary). The model learns which tokens have low reconstruction loss at early layers — i.e., which tokens your TRS diagnostic would have called stable.

**Phase 3 — Committed Routing (steps W+E onwards)**

Drop the reconstruction loss. Freeze the translator (it's done its job). Now train with only L_LM + λ_e × L_efficiency, using the router that was calibrated during exploration. The router now has a learned prior over which tokens exit early, grounded in the reconstruction signal from Phase 2.

Gradually increase λ_e over this phase — start with low efficiency pressure and increase it as the model learns to handle early-exited tokens correctly.

---

## The Key Insight Connecting All Three Parts

The reason this architecture is more principled than MoD is that the routing signal has a semantic interpretation at every stage:

- **Before training:** frequency and POS priors from your diagnostic
- **During exploration:** reconstruction quality from the translator loss
- **After exploration:** a learned router that has internalized both priors and discovered training-specific patterns

MoD's router is a black box that learns whatever minimizes loss. Your router starts informed, explores explicitly, and learns to refine a prior rather than learn from scratch. That's your architectural contribution over MoD — not just efficiency, but **interpretable, prior-informed adaptive computation.**

The final layer participation constraint (every token appears at layer N) is what makes this work practically — it ensures the model always has a chance to integrate information across exit depths and guarantees the output is well-formed regardless of routing decisions.

"""
Early Exit Transformer (EET) Architecture.

Prior-informed adaptive computation with exit-to-final-layer design.
Tokens can exit early based on semantic routing priors (frequency, POS, domain),
but ALL tokens are re-integrated at the final layer via frozen KV injection
(Option B) or masked attention (Option A).

Three-phase training:
  Phase 1 (Dense Warmup):   All tokens process all layers, router observes only.
  Phase 2 (Exploration):    Reconstruction loss + efficiency penalty, routing active.
  Phase 3 (Committed):      Fixed routing with increasing efficiency pressure.

Design grounded in Token Rank Stability (TRS) findings:
  - High-frequency tokens stabilize early (ρ = −0.32 to −0.45)
  - POS categories have consistent exit ordering
  - Code requires deeper processing than natural language
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import GPT, GPTConfig, Linear, Block, norm, has_ve


# ---------------------------------------------------------------------------
# Routing Priors
# ---------------------------------------------------------------------------

class FrequencyPrior(nn.Module):
    """Log-frequency exit bias: common tokens → early exit, rare → late.

    Checks for pre-computed freq_table.pt in tokenizer_dir. If missing,
    computes from training data and caches.
    """

    def __init__(self, vocab_size: int, tokenizer_dir: str = None, device: str = 'cpu'):
        super().__init__()
        freq_table = self._load_or_compute(vocab_size, tokenizer_dir, device)
        # Normalize to [0, 1]: 1 = most frequent (early exit), 0 = rarest
        log_freq = torch.log1p(freq_table)
        log_freq = log_freq / (log_freq.max() + 1e-8)
        self.register_buffer('freq_bias', log_freq)  # (vocab_size,)

    def _load_or_compute(self, vocab_size: int, tokenizer_dir: str = None, device: str = 'cpu') -> torch.Tensor:
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        cache_path = os.path.join(tokenizer_dir, "freq_table.pt")
        if os.path.exists(cache_path):
            print0(f"[EET] Loading frequency table from {cache_path}")
            return torch.load(cache_path, weights_only=True)

        print0(f"[EET] Computing frequency table from training data...")
        freq = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        try:
            from nanochat.dataset import resolve_data_dir, list_parquet_files
            from nanochat.tokenizer import get_tokenizer
            import pyarrow.parquet as pq
            import numpy as np
            data_dir = resolve_data_dir()
            shards = list_parquet_files(data_dir=data_dir)
            tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
            
            for idx, shard_path in enumerate(shards):
                print0(f"[EET] Processing shard {idx+1}/{len(shards)}: {os.path.basename(shard_path)}")
                table = pq.read_table(shard_path, columns=["text"])
                texts = table.column("text").to_pylist()
                texts = [t for t in texts if t]
                
                # Batched parallel tokenization (tiktoken Rust implementation)
                all_token_ids_list = tokenizer.encode(texts)
                
                # Flatten and count in PyTorch on target device (GPU)
                if all_token_ids_list:
                    flat_np = np.concatenate([np.array(tokens, dtype=np.int32) for tokens in all_token_ids_list])
                    flat_tokens_t = torch.from_numpy(flat_np).to(device=device, non_blocking=True).long()
                    freq += torch.bincount(flat_tokens_t, minlength=vocab_size)[:vocab_size]
            
            # Copy to CPU before saving to disk
            freq_cpu = freq.cpu()
            print0(f"[EET] Frequency table computed from {len(shards)} shards")
        except Exception as e:
            print0(f"[EET] Warning: could not compute freq table ({e}), using uniform")
            freq_cpu = torch.ones(vocab_size, dtype=torch.float32)

        os.makedirs(tokenizer_dir, exist_ok=True)
        torch.save(freq_cpu, cache_path)
        print0(f"[EET] Frequency table saved to {cache_path}")
        return freq_cpu

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token frequency bias. Shape: (B, T)."""
        return self.freq_bias[input_ids]


class POSPrior(nn.Module):
    """POS-category exit bias: function words → early, content words → late.

    Category mapping (exit bias, higher = earlier exit):
      punctuation/special: 1.0
      function words:      0.8
      determiners/preps:   0.6
      adjectives/adverbs:  0.4
      content words:       0.2
      numbers/symbols:     0.1
    """

    POS_EXIT_SCORES = {
        'PUNCT': 1.0, 'SPACE': 1.0, 'SYM': 0.9,
        'DET': 0.8, 'CCONJ': 0.8, 'SCONJ': 0.8, 'ADP': 0.7, 'PART': 0.7,
        'PRON': 0.6, 'AUX': 0.6, 'INTJ': 0.6,
        'ADJ': 0.4, 'ADV': 0.4,
        'VERB': 0.3, 'NOUN': 0.2, 'PROPN': 0.2,
        'NUM': 0.1, 'X': 0.3,
    }

    def __init__(self, vocab_size: int, tokenizer_dir: str = None):
        super().__init__()
        pos_table = self._load_or_compute(vocab_size, tokenizer_dir)
        self.register_buffer('pos_bias', pos_table)  # (vocab_size,)

    def _load_or_compute(self, vocab_size: int, tokenizer_dir: str = None) -> torch.Tensor:
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        cache_path = os.path.join(tokenizer_dir, "pos_categories.pt")
        if os.path.exists(cache_path):
            print0(f"[EET] Loading POS categories from {cache_path}")
            return torch.load(cache_path, weights_only=True)

        print0(f"[EET] Computing POS categories via spaCy...")
        pos_scores = torch.full((vocab_size,), 0.5, dtype=torch.float32)  # default: mid
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            except OSError:
                print0("[EET] Downloading missing spaCy model 'en_core_web_sm'...")
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            from nanochat.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)

            # Decode each token and tag it
            batch_texts = []
            batch_ids = []
            for tok_id in range(vocab_size):
                try:
                    text = tokenizer.decode([tok_id])
                    if text.strip():
                        batch_texts.append(text.strip())
                        batch_ids.append(tok_id)
                except Exception:
                    continue

            # Process in batches for efficiency
            BATCH = 1000
            for i in range(0, len(batch_texts), BATCH):
                chunk_texts = batch_texts[i:i + BATCH]
                chunk_ids = batch_ids[i:i + BATCH]
                docs = list(nlp.pipe(chunk_texts))
                for doc, tid in zip(docs, chunk_ids):
                    if len(doc) > 0:
                        pos = doc[0].pos_
                        pos_scores[tid] = self.POS_EXIT_SCORES.get(pos, 0.5)

            print0(f"[EET] POS categories computed for {len(batch_ids)} tokens")
        except Exception as e:
            print0(f"[EET] Warning: could not compute POS table ({e}), using uniform 0.5")

        os.makedirs(tokenizer_dir, exist_ok=True)
        torch.save(pos_scores, cache_path)
        print0(f"[EET] POS categories saved to {cache_path}")
        return pos_scores

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token POS exit bias. Shape: (B, T)."""
        return self.pos_bias[input_ids]


# ---------------------------------------------------------------------------
# Exit Router (3 variants)
# ---------------------------------------------------------------------------

class EarlyExitRouter(nn.Module):
    """Per-layer exit router producing per-token exit probability.

    Three architecture variants:
      'linear': Single linear projection d → 1
      'mlp1':   d → hidden → 1 (one hidden layer + ReLU)
      'mlp2':   d → hidden → hidden → 1 (two hidden layers + ReLU)

    Adds frequency and POS prior biases to the raw logit before sigmoid.
    """

    def __init__(self, n_embd: int, router_type: str = 'mlp2',
                 hidden_dim: int = 0):
        super().__init__()
        self.router_type = router_type
        hidden = hidden_dim if hidden_dim > 0 else n_embd // 4

        if router_type == 'linear':
            self.net = Linear(n_embd, 1, bias=True)
        elif router_type == 'mlp1':
            self.net = nn.Sequential(
                Linear(n_embd, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, 1, bias=True),
            )
        elif router_type == 'mlp2':
            self.net = nn.Sequential(
                Linear(n_embd, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, 1, bias=True),
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    def forward(self, h: torch.Tensor,
                freq_bias: torch.Tensor = None,
                pos_bias: torch.Tensor = None,
                freq_alpha: float = 0.0,
                pos_beta: float = 0.0) -> torch.Tensor:
        """Compute exit probability for each token.

        Args:
            h: hidden states (B, T, d)
            freq_bias: frequency prior (B, T), higher = more likely to exit
            pos_bias: POS prior (B, T), higher = more likely to exit
            freq_alpha: weight for frequency prior
            pos_beta: weight for POS prior

        Returns:
            exit_prob: (B, T) in [0, 1]
        """
        logit = self.net(h).squeeze(-1)  # (B, T)
        if freq_bias is not None and freq_alpha > 0:
            logit = logit + freq_alpha * freq_bias.to(logit.dtype)
        if pos_bias is not None and pos_beta > 0:
            logit = logit + pos_beta * pos_bias.to(logit.dtype)
        # Clamp logits to prevent sigmoid saturation — ensures gradients always
        # flow regardless of how strongly one loss pushes. sigmoid(-5)=0.007,
        # sigmoid(5)=0.993, so exit probs stay in [0.007, 0.993].
        logit = logit.clamp(-5.0, 5.0)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# TunedLens Translator
# ---------------------------------------------------------------------------

class TunedLensTranslator(nn.Module):
    """Per-layer affine translator: maps intermediate hidden state to
    final-layer prediction space for reconstruction loss.

    rank=0: full d→d linear (identity-init via zero weights)
    rank>0: d→rank→d bottleneck
    """

    def __init__(self, n_embd: int, rank: int = 0):
        super().__init__()
        if rank > 0:
            self.down = Linear(n_embd, rank, bias=False)
            self.up = Linear(rank, n_embd, bias=False)
        else:
            self.proj = Linear(n_embd, n_embd, bias=True)
        self.rank = rank

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            return self.up(self.down(h))
        return self.proj(h)


# ---------------------------------------------------------------------------
# Phase Scheduler
# ---------------------------------------------------------------------------

class EETPhaseScheduler:
    """Step-aware training phase manager for the three-phase EET schedule.

    Phase 1 (Dense Warmup):  steps [0, W)       — no routing
    Phase 2 (Exploration):   steps [W, W+E)     — reconstruction + efficiency loss
    Phase 3 (Committed):     steps [W+E, total) — increasing efficiency pressure
    """

    def __init__(self, total_steps: int, warmup_frac: float = 0.02,
                 explore_frac: float = 0.15,
                 reconstruct_lambda: float = 1.0,
                 efficiency_lambda_start: float = 0.01,
                 efficiency_lambda_end: float = 0.1):
        self.total_steps = max(total_steps, 1)
        self.warmup_end = int(warmup_frac * self.total_steps)
        self.explore_end = self.warmup_end + int(explore_frac * self.total_steps)
        self.reconstruct_lambda = reconstruct_lambda
        self.efficiency_lambda_start = efficiency_lambda_start
        self.efficiency_lambda_end = efficiency_lambda_end

    def get_phase(self, step: int) -> dict:
        """Returns current phase info.

        Keys: phase (1/2/3), do_route (bool), lambda_r (float), lambda_e (float)
        """
        if step < self.warmup_end:
            return {'phase': 1, 'do_route': False, 'lambda_r': 0.0, 'lambda_e': 0.0}
        elif step < self.explore_end:
            # Linear ramp-up of efficiency penalty during exploration, scaled by 0.01 to prevent dominating quality losses
            progress = (step - self.warmup_end) / max(self.explore_end - self.warmup_end, 1)
            lambda_e = self.efficiency_lambda_start * progress * 0.01
            return {
                'phase': 2, 'do_route': True,
                'lambda_r': self.reconstruct_lambda,
                'lambda_e': lambda_e,
            }
        else:
            # Gradually anneal efficiency up from the explore end
            progress = (step - self.explore_end) / max(self.total_steps - self.explore_end, 1)
            progress = min(progress, 1.0)
            lambda_e = (self.efficiency_lambda_start * 0.01 +
                        progress * (self.efficiency_lambda_end - self.efficiency_lambda_start * 0.01))
            return {
                'phase': 3, 'do_route': True,
                'lambda_r': 0.0,  # no reconstruction loss in committed phase
                'lambda_e': lambda_e,
            }


# ---------------------------------------------------------------------------
# EarlyExitGPT — extends GPT with early exit forward pass
# ---------------------------------------------------------------------------

class EarlyExitGPT(GPT):
    """GPT with prior-informed early exit.

    Extends GPT.__init__ to add per-layer routers, translators, and priors.
    Overrides forward() to implement masking-based early exit with frozen KV.
    """

    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__(config, pad_vocab_size_to)

        n_layer = config.n_layer
        n_embd = config.n_embd
        router_hidden = config.eet_router_hidden if config.eet_router_hidden > 0 else n_embd // 4

        # Per-layer routers (layers 0..N-2; final layer has no router)
        self.eet_routers = nn.ModuleList([
            EarlyExitRouter(n_embd, router_type=config.eet_router_type,
                            hidden_dim=router_hidden)
            for _ in range(n_layer - 1)
        ])

        # Per-layer TunedLens translators — only for 'reconstruct' variant
        if config.eet_loss_variant == 'reconstruct':
            self.eet_translators = nn.ModuleList([
                TunedLensTranslator(n_embd, rank=config.eet_translator_rank)
                for _ in range(n_layer - 1)
            ])
        else:
            self.eet_translators = nn.ModuleList()  # empty — no wasted params

        # Priors (eagerly loaded/computed in __init__ to prevent torch.compile graph breaks/OOM)
        tokenizer_dir = getattr(config, '_tokenizer_dir', None)
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        if config.eet_freq_prior_alpha > 0:
            self._freq_prior = FrequencyPrior(config.vocab_size, tokenizer_dir)
        else:
            self._freq_prior = None

        if config.eet_pos_prior_beta > 0:
            self._pos_prior = POSPrior(config.vocab_size, tokenizer_dir)
        else:
            self._pos_prior = None

    @torch.no_grad()
    def init_weights(self):
        """Initialize base GPT weights + EET-specific parameters."""
        super().init_weights()

        # Routers: small-init last layer so routers start with slight "don't exit" bias
        # but KEEP random weights so the output varies across tokens from step 1.
        # Zero-ing the weights causes the router to collapse to a constant function.
        for router in self.eet_routers:
            if router.router_type == 'linear':
                # Scale down weights for stable init, neutral bias
                nn.init.normal_(router.net.weight, std=0.01)
                nn.init.constant_(router.net.bias, 0.0)  # sigmoid(0) = 0.5 — neutral start
            else:
                # Scale down (not zero!) last linear in MLP chain
                last_linear = list(router.net.modules())[-1]
                if isinstance(last_linear, (Linear, nn.Linear)):
                    nn.init.normal_(last_linear.weight, std=0.01)
                    nn.init.constant_(last_linear.bias, 0.0)  # sigmoid(0) = 0.5 — neutral start

        # Translators: identity-like init (only when present)
        for translator in self.eet_translators:
            if translator.rank > 0:
                nn.init.zeros_(translator.down.weight)
                nn.init.zeros_(translator.up.weight)
            else:
                nn.init.zeros_(translator.proj.weight)
                nn.init.zeros_(translator.proj.bias)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                eet_do_route=False, eet_phase=1, eet_lambda_r=0.0, eet_lambda_e=0.0):
        """Forward pass with early exit routing.

        Phase scheduling is done OUTSIDE this method (in base_train.py) to
        avoid torch.compile creating a single graph covering all phases.
        Args eet_do_route (bool) and eet_phase (int) are stable compile guards
        that cause at most 3 compiled graph variants. eet_lambda_r/e are Python
        floats converted to scalar tensors inside to avoid per-step recompiles.
        """
        B, T = idx.size()
        config = self.config

        do_route = eet_do_route and self.training

        # Phase 1 (no routing): delegate to parent GPT.forward() to get the
        # exact same torch.compile graph as dense — avoiding OOM from the
        # compiler generating a less memory-efficient graph for the EET
        # forward's additional dead-code paths (frozen_h, reconstruction, etc.)
        if not do_route:
            loss_or_logits = super().forward(idx, targets, kv_cache, loss_reduction)
            if self.training and targets is not None:
                self._eet_diagnostics = {
                    'phase': eet_phase,
                    'active_frac': torch.tensor(1.0, device=idx.device),
                    'total_exit_frac': torch.tensor(0.0, device=idx.device),
                }
            return loss_or_logits

        # --- Standard GPT embedding ---
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T
        if T_total > self.cos.size(1):
            new_len = max(T_total, self.cos.size(1) * 2)
            head_dim = config.n_embd // config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total]

        if self.embedding_model is None:
            x = self.transformer.wte(idx)
        else:
            x, _ = self.embedding_model(idx)
        if "wpe" in self.transformer:
            positions = torch.arange(T0, T_total, device=idx.device)
            x = x + self.transformer.wpe(positions)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x  # save for x0 residual

        freq_bias = self._freq_prior(idx) if self._freq_prior is not None else None
        pos_bias = self._pos_prior(idx) if self._pos_prior is not None else None

        # --- Early exit forward pass ---
        is_soft_training = eet_do_route and self.training and eet_phase == 2

        token_active = torch.ones(B, T, dtype=torch.bool, device=x.device)
        frozen_h = torch.zeros_like(x)
        reconstruction_losses = []
        total_exit_frac = torch.tensor(0.0, device=x.device)
        n_routed_layers = 0

        loss_variant = config.eet_loss_variant
        blocks = list(self.transformer.h)
        n_layer = len(blocks)
        need_exit_tracking = (eet_phase == 2 and loss_variant != 'reconstruct')
        
        if need_exit_tracking:
            exit_hidden = torch.zeros_like(x)                # (B, T, D)
            exit_layers = torch.full((B, T), n_layer - 1, dtype=torch.long, device=x.device)
            if loss_variant == 'entropy_surprise':
                prev_exit_hidden = torch.zeros_like(x)       # for surprise term
                
        if is_soft_training:
            exit_probs_list = []
            candidate_states = []
            candidate_prev_states = []
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))

        prev_x = x.detach()  # hidden state before first block

        for i, block in enumerate(blocks):
            # Apply per-layer residual scaling
            x0_w = self.x0_lambdas[i]
            if self._use_residual_decay and self.depth_decay_raw is not None:
                decay_base = torch.sigmoid(self.depth_decay_raw)
                x0_w = x0_w * (decay_base ** i)
            x_input = self.resid_lambdas[i] * x + x0_w * x0
            
            prev_x_val = x_input  # track state before block

            ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
            x_new = block(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)

            if is_soft_training:
                # Dense forward with residual mixing for soft training
                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x_new.dtype)
                    mixed = self.residual_mixers[i](x_new.transpose(1, 2)).transpose(1, 2)
                    x_new = x_new + gamma * mixed
                x = x_new
                
                # Routing check for soft path
                if i in routing_layers:
                    exit_prob_i = self.eet_routers[i](
                        x.detach() if self.training else x,
                        freq_bias=freq_bias, pos_bias=pos_bias,
                        freq_alpha=config.eet_freq_prior_alpha,
                        pos_beta=config.eet_pos_prior_beta,
                    )
                    exit_probs_list.append(exit_prob_i)
                    candidate_states.append(norm(x))
                    candidate_prev_states.append(norm(prev_x_val))
                elif i == n_layer - 1:
                    candidate_states.append(norm(x))
                    candidate_prev_states.append(norm(prev_x_val))
            else:
                # --- Hard Early Exit path (Phase 3 / Eval / Inference) ---
                if do_route:
                    active_mask = token_active.unsqueeze(-1)  # (B, T, 1)
                    x = torch.where(active_mask, x_new, frozen_h)
                else:
                    x = x_new

                # Routing decision
                if i < n_layer - 1 and do_route and i >= config.eet_min_exit_layer:
                    exit_prob = self.eet_routers[i](
                        x.detach() if self.training else x,
                        freq_bias=freq_bias, pos_bias=pos_bias,
                        freq_alpha=config.eet_freq_prior_alpha,
                        pos_beta=config.eet_pos_prior_beta,
                    )  # (B, T)

                    exit_candidates = (exit_prob > config.eet_exit_threshold) & token_active
                    n_max_exits = int(config.eet_max_frozen_kv_frac * B * T)
                    probs_for_cap = exit_prob * exit_candidates.float()
                    _, top_idx = probs_for_cap.view(-1).topk(n_max_exits, largest=True, sorted=False)
                    budget_mask = torch.zeros(B * T, dtype=torch.bool, device=x.device)
                    budget_mask.scatter_(0, top_idx, True)
                    exit_mask = exit_candidates & budget_mask.view(B, T)

                    frozen_h = torch.where(
                        exit_mask.unsqueeze(-1),
                        x.detach(),
                        frozen_h,
                    )
                    token_active = token_active & ~exit_mask
                    total_exit_frac = total_exit_frac + exit_mask.float().mean()
                    n_routed_layers += 1

                    if eet_phase == 2:
                        if loss_variant == 'reconstruct':
                            translated = self.eet_translators[i](x.detach())
                            reconstruction_losses.append((translated, exit_mask.float()))
                        elif need_exit_tracking:
                            exit_hidden = torch.where(exit_mask.unsqueeze(-1), x.detach(), exit_hidden)
                            exit_layers = torch.where(exit_mask, torch.tensor(i, device=x.device), exit_layers)
                            if loss_variant == 'entropy_surprise':
                                prev_exit_hidden = torch.where(exit_mask.unsqueeze(-1), prev_x, prev_exit_hidden)

                if need_exit_tracking:
                    prev_x = x.detach()

                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x.dtype)
                    mixed = self.residual_mixers[i](x.transpose(1, 2)).transpose(1, 2)
                    x = x + gamma * mixed

        # --- Final post-processing ---
        if is_soft_training:
            p_exits = []
            p_reach = torch.ones(B, T, dtype=x.dtype, device=x.device)
            soft_h = torch.zeros_like(x)
            soft_prev_h = torch.zeros_like(x)
            soft_active = torch.zeros(B, T, dtype=x.dtype, device=x.device)

            for idx, (r_i, h_i, prev_h_i) in enumerate(zip(exit_probs_list, candidate_states[:-1], candidate_prev_states[:-1])):
                p_exit_i = r_i * p_reach
                p_exits.append(p_exit_i)
                # Build blended states with LIVE exit probs (gradient flows to router
                # through auxiliary losses: entropy_surprise, adversarial, reconstruct).
                # The LM loss path is blocked by detaching at `x = soft_h.detach()` below.
                soft_h = soft_h + p_exit_i.unsqueeze(-1) * h_i
                soft_prev_h = soft_prev_h + p_exit_i.unsqueeze(-1) * prev_h_i
                # Efficiency path: live p_exit_i
                soft_active = soft_active + p_exit_i * ((config.eet_min_exit_layer + idx + 1) / n_layer)
                
                if loss_variant == 'reconstruct':
                    block_idx = config.eet_min_exit_layer + idx
                    translated = self.eet_translators[block_idx](h_i.detach())
                    reconstruction_losses.append((translated, p_exit_i))  # live p_exit_i
                    
                p_reach = p_reach * (1.0 - r_i)

            soft_h = soft_h + p_reach.unsqueeze(-1) * candidate_states[-1]
            soft_prev_h = soft_prev_h + p_reach.unsqueeze(-1) * candidate_prev_states[-1]
            soft_active = soft_active + p_reach * 1.0
            p_exits.append(p_reach)
            
            # Save stacked soft exit probabilities for structure diagnostics
            self._last_exit_probs = torch.stack(p_exits, dim=-1)

            # DETACH here: LM loss (logits → CE) trains the backbone only, not the router.
            # Auxiliary losses (entropy_surprise, adversarial, reconstruct) use exit_hidden
            # which keeps the live gradient path to the router.
            x = soft_h.detach()
            exit_hidden = soft_h          # live — aux losses backprop to router
            prev_exit_hidden = soft_prev_h  # live — aux losses backprop to router
            avg_active = soft_active.mean()
            total_exit_frac = 1.0 - avg_active
            n_routed_layers = len(routing_layers)
        else:
            if do_route:
                x = torch.where(token_active.unsqueeze(-1), x, frozen_h)

            if need_exit_tracking:
                never_exited = token_active.unsqueeze(-1)
                exit_hidden = torch.where(never_exited, x.detach(), exit_hidden)
                if loss_variant == 'entropy_surprise':
                    prev_exit_hidden = torch.where(never_exited, prev_x, prev_exit_hidden)

            x = norm(x)
            avg_active = token_active.float().mean()

        # --- LM head ---
        softcap = 20
        logits = self.lm_head(x)
        logits = logits[..., :config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )

            # --- Phase 2 variant losses ---
            if eet_phase == 2 and loss_reduction == 'mean':
                if loss_variant == 'reconstruct' and reconstruction_losses:
                    recon_loss_term = self._compute_reconstruction_loss(
                        logits, reconstruction_losses, eet_lambda_r, config.vocab_size
                    )
                    loss = loss + recon_loss_term
                elif loss_variant == 'entropy_surprise' and need_exit_tracking:
                    variant_loss, variant_diag = self._compute_entropy_surprise_loss(
                        exit_hidden, prev_exit_hidden, config
                    )
                    loss = loss + variant_loss
                elif loss_variant == 'adversarial' and need_exit_tracking:
                    variant_loss, variant_diag = self._compute_adversarial_loss(
                        exit_hidden, logits, targets, config
                    )
                    loss = loss + variant_loss

            # --- Efficiency loss (Phase 2 & 3) ---
            if (do_route or is_soft_training) and n_routed_layers > 0 and loss_reduction == 'mean':
                loss = loss + eet_lambda_e * avg_active

            # Store diagnostics
            if self.training:
                self._eet_diagnostics = {
                    'phase': eet_phase,
                    'active_frac': avg_active,
                    'total_exit_frac': total_exit_frac if is_soft_training else (total_exit_frac / max(n_routed_layers, 1)),
                }

            return loss
        else:
            return logits

    @torch.compiler.disable
    def _compute_reconstruction_loss(self, logits, reconstruction_losses, eet_lambda_r, vocab_size):
        recon_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        target_logits = logits.detach().view(-1, logits.size(-1))
        for translated_h, exit_weight in reconstruction_losses:
            flat_mask = (exit_weight > 0).view(-1)
            
            # Index only exited tokens
            exited_h = translated_h.view(-1, translated_h.size(-1))[flat_mask]
            exited_target_logits = target_logits[flat_mask]
            flat_weights = exit_weight.view(-1)[flat_mask]
            
            # Project only exited tokens to vocabulary size (saving huge memory)
            trans_logits = self.lm_head(norm(exited_h))
            trans_logits = trans_logits[..., :vocab_size].float()
            
            # Compute KL divergence on exited tokens
            t_probs = F.softmax(exited_target_logits, dim=-1)
            t_log_probs = F.log_softmax(exited_target_logits, dim=-1)
            e_log_probs = F.log_softmax(trans_logits, dim=-1)
            kl_exited = (t_probs * (t_log_probs - e_log_probs)).sum(-1)
            recon_loss = recon_loss + (kl_exited * flat_weights).sum() / flat_weights.sum().clamp(min=1e-5)
            
        return eet_lambda_r * recon_loss / max(len(reconstruction_losses), 1)

    @torch.compiler.disable
    def _compute_entropy_surprise_loss(self, exit_hidden, prev_exit_hidden, config):
        """Variant A: Entropy + Surprise loss (eager mode).

        Entropy: topk approximation of vocab entropy at exit-layer hidden states.
                 Low entropy = confident → safe to exit.
        Surprise: relative norm of update between exit layer and previous layer.
                  High surprise = representation still changing → unsafe to exit.

        Returns (loss_term, diagnostics_dict).
        """
        # exit_hidden: (B, T, D) — hidden state at each token's exit layer
        # prev_exit_hidden: (B, T, D) — hidden state at layer before exit
        B, T, D = exit_hidden.shape
        topk_vocab = config.eet_topk_vocab
        unembed = self.lm_head.weight  # (padded_vocab, D)

        # --- Entropy: chunked projection to avoid (B*T, V) peak ---
        flat_h = exit_hidden.reshape(-1, D)  # (N, D)
        N = flat_h.shape[0]
        chunk_size = 8192  # ~1 GB per chunk for V=32768 float32
        entropies = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # Raw projection — intentionally no layernorm (cheaper, different signal)
            chunk_logits = flat_h[start:end].float() @ unembed[:config.vocab_size].float().T  # (chunk, V)
            topk_logits = chunk_logits.topk(topk_vocab, dim=-1).values  # (chunk, topk)
            probs = F.softmax(topk_logits, dim=-1)
            log_probs = F.log_softmax(topk_logits, dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)  # (chunk,)
            entropies.append(ent)
        entropy_loss = torch.cat(entropies).mean()

        # --- Surprise: relative norm of representation update ---
        update = exit_hidden - prev_exit_hidden  # (B, T, D)
        surprise = update.norm(dim=-1) / (prev_exit_hidden.norm(dim=-1) + 1e-8)  # (B, T)
        surprise_loss = surprise.mean()

        # Entropy: minimize (exit when confident)
        # Surprise: SUBTRACT (penalize exiting when representation still changing)
        loss_term = (
            config.eet_entropy_lambda * entropy_loss
            - config.eet_surprise_lambda * surprise_loss
        )

        diag = {
            'entropy': entropy_loss.detach(),
            'surprise': surprise_loss.detach(),
        }
        return loss_term, diag

    @torch.compiler.disable
    def _compute_adversarial_loss(self, exit_hidden, full_logits, targets, config):
        """Variant B: Adversarial gap + Entropy stabilizer (eager mode).

        Adversarial: CE gap between early-exit predictions and full-depth predictions.
                     Router minimizes quality degradation from early exit.
        Entropy: same topk entropy as Variant A — stabilizes adversarial signal early.

        Uses single forward pass: exit_hidden from the loop, full_logits already computed.
        Returns (loss_term, diagnostics_dict).
        """
        B, T, D = exit_hidden.shape
        topk_vocab = config.eet_topk_vocab
        unembed = self.lm_head.weight  # (padded_vocab, D)
        vocab_size = config.vocab_size
        softcap = 20

        # --- Adversarial: early-exit LM loss vs full-depth LM loss ---
        # Compute early-exit logits via chunked norm + lm_head projection
        flat_h = exit_hidden.reshape(-1, D)  # (N, D)
        N = flat_h.shape[0]
        chunk_size = 8192
        # CE losses
        flat_targets = targets.reshape(-1)  # (B*T,)
        loss_full = F.cross_entropy(
            full_logits.detach().view(-1, full_logits.size(-1)),
            flat_targets, ignore_index=-1,
        )

        # Compute unreduced cross entropy chunk-by-chunk to save massive peak memory (avoiding B*T*V tensor)
        total_ce_early = torch.tensor(0.0, device=exit_hidden.device, dtype=torch.float32)
        total_valid = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_normed = norm(flat_h[start:end])  # apply same layernorm as real forward
            chunk_logits = self.lm_head(h_normed)  # (chunk, padded_V)
            chunk_logits = chunk_logits[..., :vocab_size].float()
            chunk_logits = softcap * torch.tanh(chunk_logits / softcap)
            
            chunk_targets = flat_targets[start:end]
            chunk_ce = F.cross_entropy(chunk_logits, chunk_targets, ignore_index=-1, reduction='sum')
            total_ce_early = total_ce_early + chunk_ce
            total_valid += (chunk_targets != -1).sum().item()
        
        loss_early = total_ce_early / max(total_valid, 1)

        # Gap: how much worse is early exit vs full depth? Clamp at 0 (no reward for being better)
        adversarial_loss = (loss_early - loss_full).clamp(min=0)

        # --- Entropy stabilizer (same as Variant A, cheap signal) ---
        entropies = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_logits = flat_h[start:end].float() @ unembed[:vocab_size].float().T
            topk_logits = chunk_logits.topk(topk_vocab, dim=-1).values
            probs = F.softmax(topk_logits, dim=-1)
            log_probs = F.log_softmax(topk_logits, dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)
            entropies.append(ent)
        entropy_loss = torch.cat(entropies).mean()

        loss_term = (
            config.eet_adv_lambda * adversarial_loss
            + config.eet_adv_entropy_lambda * entropy_loss
        )

        diag = {
            'adversarial': adversarial_loss.detach(),
            'entropy': entropy_loss.detach(),
            'lm_early': loss_early.detach(),
            'lm_full': loss_full.detach(),
        }
        return loss_term, diag

    def num_scaling_params(self):
        """Override to include EET params in scaling count without causing assertion errors."""
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters()) if "wpe" in self.transformer else 0
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        # EET: Add router and translator parameters to transformer_matrices (and keep track of them separately)
        eet_params = sum(p.numel() for p in self.eet_routers.parameters())
        eet_params += sum(p.numel() for p in self.eet_translators.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters()) + eet_params

        research = 0
        if self.embedding_model is not None:
            research += sum(p.numel() for p in self.embedding_model.parameters())
        if self.aux_head is not None:
            research += sum(p.numel() for p in self.aux_head.parameters())

        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        if self.depth_decay_raw is not None:
            scalars += self.depth_decay_raw.numel()
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                scalars += gamma_p.numel()
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                scalars += sum(p.numel() for p in mixer.parameters())

        total = wte + wpe + value_embeds + lm_head + transformer_matrices + research + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch in EarlyExitGPT"
        return {
            'wte': wte,
            'wpe': wpe,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'research': research,
            'scalars': scalars,
            'eet_routers_and_translators': eet_params,
            'total': total,
        }

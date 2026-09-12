"""O6: does a natively binary transformer train from scratch at all?

NOT A RESULT, and it is not permitted to be quoted as one. Depth 2 to 4, a few
million tokens. It separates "trains" from "diverges", exercises every seam in
section 4, and gives the flip-rate and activation-balance instrumentation something
to report before those metrics matter at scale.

The gate: loss goes down and no layer saturates. If it diverges, the plan names the
first two suspects in order, residual accumulator width and threshold initialisation.

Three diagnostics, because a binary network fails in ways a loss curve hides:
  flip rate       fraction of latent weights whose SIGN changed this step. A rule
                  that never flips and one that flips constantly both look like
                  convergence failure in the loss and are distinguishable only here.
  activation      fraction of +1 in each layer's binarised input. A layer stuck near
  balance         0 or 1 is dead and stays invisible in the loss for a long time.
  dead bits       latent weights parked outside the STE clip window, which therefore
                  receive no gradient and can never flip again.

Run:  python -m scripts.o6_binary_smoke --steps 400
"""
import argparse

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.binary import BinaryLinear, BinaryEmbedding, binarise_model_
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--arch", default="binary", choices=["binary", "dense", "both"])
    ap.add_argument("--no-binarise-acts", action="store_true",
                    help="W1A16 rung: weights binary, activations fp")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer("tokenizer")
    token_bytes = get_token_bytes(device=dev, tokenizer_dir="tokenizer")
    n_embd = 128 * max(1, a.depth * 64 // 128)

    arches = ["dense", "binary"] if a.arch == "both" else [a.arch]
    for arch in arches:
        torch.manual_seed(0)
        cfg = GPTConfig(sequence_len=a.seq, vocab_size=a.vocab, n_layer=a.depth,
                        n_head=n_embd // 128, n_kv_head=n_embd // 128, n_embd=n_embd,
                        window_pattern="L")
        with torch.device("meta"):
            model = GPT(cfg)
        model.to_empty(device=dev)
        model.init_weights()
        if arch == "binary":
            sw = binarise_model_(model, binarise_acts=not a.no_binarise_acts)
            print(f"binarised {len(sw)} modules "
                  f"({sum(1 for _, k in sw if k=='Linear')} Linear, "
                  f"{sum(1 for _, k in sw if k=='Embedding')} Embedding)")

        bmods = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, (BinaryLinear, BinaryEmbedding))]
        balance = {}

        def mk(name):
            def hook(mod, args):
                if args and torch.is_tensor(args[0]):
                    balance[name] = float((args[0] > 0).to(torch.float32).mean())
            return hook

        handles = [m.register_forward_pre_hook(mk(n))
                   for n, m in bmods if isinstance(m, BinaryLinear)]

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tok, a.batch, a.seq, split="train", device=dev, data_dir=a.data_dir)

        prev = {n: torch.sign(m.weight.detach()).clone() for n, m in bmods}
        print(f"\n=== {arch.upper()}  depth={a.depth} d={n_embd} V={a.vocab} ===")
        print(f"{'step':>6}{'loss':>10}{'flip%':>9}{'bal min':>9}{'bal max':>9}{'dead%':>8}")
        losses = []
        for step in range(a.steps):
            x, y = next(loader)
            out = model(x, targets=y)
            loss = out[0] if isinstance(out, tuple) else out
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if step % a.log_every == 0 or step == a.steps - 1:
                flips = dead = tot = 0
                for n, m in bmods:
                    cur = torch.sign(m.weight.detach())
                    flips += int((cur != prev[n]).sum())
                    dead += int((m.weight.detach().abs() > m.clip).sum())
                    tot += m.weight.numel()
                    prev[n] = cur.clone()
                bmin = min(balance.values()) if balance else float("nan")
                bmax = max(balance.values()) if balance else float("nan")
                print(f"{step:>6}{sum(losses[-a.log_every:])/max(1,len(losses[-a.log_every:])):>10.4f}"
                      f"{100*flips/max(tot,1):>9.3f}{bmin:>9.3f}{bmax:>9.3f}"
                      f"{100*dead/max(tot,1):>8.2f}")
        for h in handles:
            h.remove()

        model.eval()
        val = tokenizing_distributed_data_loader_bos_bestfit(
            tok, a.batch, a.seq, split="val", device=dev, data_dir=a.data_dir)
        bpb, _ = evaluate_bpb(model, val, a.eval_steps, token_bytes)
        first = sum(losses[:20]) / 20
        last = sum(losses[-20:]) / 20
        print(f"final val bpb {float(bpb):.4f}   loss {first:.4f} -> {last:.4f}")
        gate = "PASS" if last < first - 0.05 else "FAIL"
        print(f"GATE (loss went down): {gate}")
        if arch == "binary":
            if bmax > 0.95 or bmin < 0.05:
                print("GATE (no saturated layer): FAIL - a layer is stuck at one sign.")
                print("  Suspects in order: residual accumulator width, then threshold init.")
            else:
                print("GATE (no saturated layer): PASS")
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

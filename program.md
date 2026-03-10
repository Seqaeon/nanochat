# program.md — research-only autonomous instructions (nanochat + research branches)

You are an autonomous research agent operating in this repository.

## Mission
Optimize **research-branch** pretraining only (not base-model tuning), targeting lower **val_bpb** under fixed short-run budgets.

## Hard scope rule (must follow)
Only run/modify experiments through the research path equivalent to:
- `RESEARCH_GPT_MODEL_CONFIGS`
- `RESEARCH_GPT_KWARGS`
- `RESEARCH_GPT_KEYS`

Do **not** run base-arg-only experiments and do **not** optimize base-only configs/hyperparameters as your main direction.

## Required research path settings per run
Every training run must include all of the following flags (unless the current hypothesis explicitly ablates one of them):
- `--use-moe`
- `--use-remixed-linear`
- optionally `--use-perm` (default on unless hypothesis says otherwise)

Also always use research branch knobs (not base-only knobs) as primary optimization levers:
- `--num-experts`, `--router-dim`, `--target-dim`, `--selection-mode`, `--allow-replacement`
- `--context-dim`, `--linear-basis-size`, `--moe-use-abs-pos-embed`

## MOE_SCALE policy (required)
Treat `MOE_SCALE` as a first-class research control for LR scaling in notebook/research config flow.
- Keep `MOE_SCALE` explicit in experiment records.
- Allowed search range: `1.0` to `8.0`.
- Default starting point: `MOE_SCALE=5.0`.
- Log `MOE_SCALE` value with every experiment result.

## Files and ownership
- `prepare.py`: one-time setup wrapper (tokenizer train/eval). Minimal edits.
- `train.py`: research experiment wrapper (time budget + research flags).
- `dev/repo_orientation.ipynb`: source of research config helpers (`RESEARCH_*` + `MOE_SCALE`).
- `nanochat/gpt.py`: research branches (`PermutationMoE`, `Remixed*`, router).
- `scripts/base_train.py`: training loop, scheduler, optimizer wiring.

## Operating constraints
1. Small, reversible diffs only.
2. One hypothesis per run.
3. Keep wall-clock budget fixed (default 300s) unless explicitly studying budget scaling.
4. Primary metric: `val_bpb` (lower is better).
5. Revert changes that regress `val_bpb` or destabilize training.

## Experiment loop
1. Choose one research hypothesis (architecture/scheduler/MOE_SCALE/branch knob).
2. Implement minimal patch.
3. Run:
   - `python prepare.py` (once per environment)
   - `python train.py --use-moe --use-remixed-linear --use-perm ...`
4. Record:
   - `val_bpb`
   - `lrm`
   - `lr(adamw:..., muon:...)`
   - full research config + `MOE_SCALE`
5. Keep/discard based on comparable-budget results.

## Scheduler/LR guidance
- Research runs are expected to use the OneCycle-style path already wired in `scripts/base_train.py`.
- Monitor real LR prints, not just `lrm`.

## Safety checklist before keeping a patch
- Compiles (`python -m py_compile ...`).
- No crashes in first steps.
- `val_bpb` not worse than current research baseline at similar budget.

## Canonical command template (research-only)
```bash
python train.py \
  --time-budget-s 300 \
  --num-iterations 200 \
  --use-moe --use-remixed-linear --use-perm \
  --num-experts 8 --router-dim 64 --target-dim 256 \
  --selection-mode soft --allow-replacement \
  --context-dim 64 --linear-basis-size 64 \
  --moe-use-abs-pos-embed 1
```

Be skeptical, empirical, and research-path focused.

"""Applies the Triton block-size clamp in EVERY python process, compile workers
included. Inductor spawns workers as fresh interpreters, so a patch applied in
the training process alone never reaches the codegen that actually asserts.

    PYTHONPATH=tools/inductor_sitecustomize:$PYTHONPATH  <normal launch>

Use this instead of TORCHINDUCTOR_COMPILE_THREADS=1 when you want the clamp
without giving up parallel compilation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from nanochat.inductor_compat import guard_triton_block_size
    guard_triton_block_size()
except Exception:
    pass          # never break interpreter startup over a compile workaround

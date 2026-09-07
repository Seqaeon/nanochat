"""The tracked tokenizer must be the pinned one.

`tokenizer/` is committed so that a `git pull` REPAIRS a box sitting on a bad copy.
Untracking it would instead delete the tokenizer on every box whose copy is clean, or
refuse the pull outright on a box whose copy differs.

The cost of tracking it is that anything writing into `./tokenizer` gets swept into the
next commit. That happened: `tokenizer.pkl` flip-flopped between the real 412 KB V=32768
file and a 1.9 KB V=265 byte-level stub across five commits. A run against the stub does
not fail. base_train prints "Vocab size: 265", pads to 320 and trains to completion, so a
whole sweep produces bpb numbers that compare with nothing, and the EET P02 dense controls
were thrown away because of it.

These tests are the tripwire. Run them before committing anything that touched the
tokenizer directory.
"""
import hashlib
import json
import os

import pytest
import torch

TOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer")
PIN_PATH = os.path.join(TOK_DIR, "PINNED.json")


def _pin():
    if not os.path.exists(PIN_PATH):
        pytest.skip(f"no pin file at {PIN_PATH}")
    with open(PIN_PATH) as f:
        return json.load(f)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_tracked_tokenizer_matches_the_pin():
    """Every tracked tokenizer file must hash to the pinned value.

    A mismatch means something overwrote the tokenizer in place. Do not commit it: work
    out what wrote there, restore with `git checkout -- tokenizer/`, and only update
    PINNED.json when you are deliberately changing which tokenizer the repo ships.
    """
    pin = _pin()
    wrong = []
    for name, want in pin["sha256"].items():
        path = os.path.join(TOK_DIR, name)
        if not os.path.exists(path):
            wrong.append(f"{name}: MISSING")
            continue
        got = _sha(path)
        if got != want:
            wrong.append(f"{name}: {got[:16]}... != pinned {want[:16]}... "
                         f"({os.path.getsize(path)} bytes)")
    assert not wrong, (
        "tokenizer/ does not match tokenizer/PINNED.json:\n  "
        + "\n  ".join(wrong)
        + "\n\nRestore with: git checkout -- tokenizer/\n"
          "Only regenerate PINNED.json when deliberately shipping a different tokenizer."
    )


def test_tracked_tokenizer_is_not_a_stub():
    """Independent of the hash: the vocabulary must not be byte-level.

    This is the check that would have caught the V=265 stub even if someone had
    regenerated the pin file along with it.
    """
    from nanochat.tokenizer import get_tokenizer

    pin = _pin()
    vocab = get_tokenizer(TOK_DIR).get_vocab_size()
    assert vocab >= 1000, (
        f"tokenizer/ has vocab_size={vocab}, which is the byte-level stub. base_train "
        f"will train against it without complaining and every bpb it produces will be "
        f"incomparable with the rest of the repo's results."
    )
    assert vocab == pin["vocab_size"], f"vocab_size={vocab}, pinned {pin['vocab_size']}"


@pytest.mark.parametrize("name", ["freq_table.pt", "token_bytes.pt"])
def test_side_tables_match_the_vocabulary(name):
    """A table sized for a different vocabulary silently changes what it indexes.

    `FrequencyPrior` reads freq_table.pt, and `evaluate_bpb` reads token_bytes.pt. A stale
    one from another vocabulary either indexes out of range or, worse, does not.
    """
    pin = _pin()
    t = torch.load(os.path.join(TOK_DIR, name), weights_only=True)
    assert t.numel() == pin["vocab_size"], (
        f"{name} has {t.numel()} entries but the tokenizer has {pin['vocab_size']}. "
        f"Rebuild it rather than letting it index the wrong vocabulary."
    )

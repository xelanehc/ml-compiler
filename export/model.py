"""The one definition of "the model" that every part of the project shares.

The exporter, the golden-output generator, and all three baselines import from
here, so they are guaranteed to be running the same network on the same inputs.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
GOLDEN = ARTIFACTS / "golden"

# Sequence lengths we compile and benchmark. One static-shape build per entry.
SEQ_LENS = (16, 64, 256, 1024)

VOCAB_SIZE = 50257
INPUT_SEED = 0


class GPT2Logits(torch.nn.Module):
    """GPT-2 small as a pure function: token ids (1, T) int64 -> logits (1, T, 50257) fp32.

    The Hugging Face model returns a rich output object and can return a
    key/value cache. A compiler wants one tensor in and one tensor out, so this
    wrapper pins the signature down and turns the cache off.
    """

    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids):
        return self.gpt2(input_ids=input_ids, use_cache=False).logits


def load_model():
    return GPT2Logits().eval()


def make_input_ids(seq_len):
    """Deterministic token ids for a given length.

    Timing does not depend on which tokens we feed, only on how many, so
    seeded random ids are enough. Using the same seed everywhere means every
    system is checked against the same golden logits.
    """
    rng = np.random.default_rng(INPUT_SEED + seq_len)
    return rng.integers(0, VOCAB_SIZE, size=(1, seq_len), dtype=np.int64)


def golden_paths(seq_len):
    return GOLDEN / f"T{seq_len}_ids.npy", GOLDEN / f"T{seq_len}_logits.npy"


def load_golden(seq_len):
    ids_path, logits_path = golden_paths(seq_len)
    return np.load(ids_path), np.load(logits_path)

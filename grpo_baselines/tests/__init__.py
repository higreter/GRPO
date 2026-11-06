"""CPU test suite for the four GRPO baseline implementations.

Run with the standard library only (the read-only ``judgelrm-sft`` environment has no
pytest)::

    python -m unittest discover -s tests -t . -v

Every test here runs without a GPU, without ``trl``/``verl``/``datasets`` and without
touching the model weights.  That is deliberate: the parts of these baselines that can
be silently wrong -- label mapping, prompt construction, advantage arithmetic, reward
direction -- are all pure functions, so they can be verified exhaustively on CPU.  What
these tests can *not* establish is that a real training run converges; the report's
status table says so explicitly.
"""

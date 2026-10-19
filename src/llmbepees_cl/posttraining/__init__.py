"""Post-training (Fase 6): DPO from-scratch untuk memahami preference tuning.

Reusable dan diuji di `tests/test_posttraining_dpo.py`. Skrip belajar + versi Colab
(model open-weight) ada di `06-specialization/`.
"""

from llmbepees_cl.posttraining.data import PREFERENCE_PAIRS, PreferencePair
from llmbepees_cl.posttraining.dpo import dpo_accuracy, dpo_loss, sequence_logprobs
from llmbepees_cl.posttraining.train import encode_prompt_response, evaluate_dpo, train_dpo

__all__ = [
    "PREFERENCE_PAIRS",
    "PreferencePair",
    "dpo_accuracy",
    "dpo_loss",
    "encode_prompt_response",
    "evaluate_dpo",
    "sequence_logprobs",
    "train_dpo",
]

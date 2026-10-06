"""Loading Hugging Face models quietly, while still refusing checkpoints with missing weights."""

from transformers import AutoModel
from transformers.utils import logging

# Weights the library never uses, which some embedding checkpoints leave out.
UNUSED_WEIGHTS = ("pooler.",)


def load_model(name: str, **options):
    """Load the base model of a checkpoint without printing transformers' load report.

    A causal LM's checkpoint also holds its output head, which the base model does not use; the
    report would list it as unexpected. Weights the base model needs but the checkpoint lacks are an
    error.
    """
    verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    try:
        model, info = AutoModel.from_pretrained(name, output_loading_info=True, **options)
    finally:
        logging.set_verbosity(verbosity)
    missing = sorted(key for key in info["missing_keys"] if not key.startswith(UNUSED_WEIGHTS))
    if missing or info["mismatched_keys"] or info["error_msgs"]:
        problems = missing or info["mismatched_keys"] or info["error_msgs"]
        raise RuntimeError(f"{name} did not load completely: {list(problems)[:5]}")
    return model

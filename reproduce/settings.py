"""The paper's models and settings (Appendix A.1).

One setting per (LLM, embedder) pair, never tuned per task, evaluated on the full sets: every pair
relays with the multiply form, strength 1 and the mass-preserving bias, into every layer of an
embedder with CLS or last-token pooling. Every model is pinned to the revision every run used.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    name: str
    repository: str
    revision: str
    n_blocks: int
    block: int
    """The block read, counted from 0, set by the label-free rule (attention_relay.choose_block)."""
    template_from: str | None = None
    """For a base checkpoint: its instruction-tuned sibling, whose chat template frames prompts."""


@dataclass(frozen=True)
class EmbedderSettings:
    name: str
    repository: str
    revision: str
    pooling: str
    """"mean", "cls" or "last"."""
    max_tokens: int | None
    """Texts are truncated to this many of the embedder's tokens; None keeps them whole."""
    prefix: str = ""
    code_revision: str | None = None
    """For models with custom code: the revision of that code."""


LLMS = {
    "qwen3-0.6b": LLMSettings(
        "Qwen3-0.6B", "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 28, 24
    ),
    "qwen3-1.7b": LLMSettings(
        "Qwen3-1.7B", "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 28, 24
    ),
    "qwen3-4b": LLMSettings(
        "Qwen3-4B", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 36, 29
    ),
    "qwen3-8b": LLMSettings(
        "Qwen3-8B", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", 36, 29
    ),
    "llama-3.1-8b-instruct": LLMSettings(
        "Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
        32,
        24,
    ),
    "olmo-3-7b-instruct": LLMSettings(
        "OLMo-3-7B-Instruct",
        "allenai/Olmo-3-7B-Instruct",
        "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        32,
        23,
    ),
    "qwen3-1.7b-base": LLMSettings(
        "Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        28,
        24,
        template_from="qwen3-1.7b",
    ),
    "qwen3-8b-base": LLMSettings(
        "Qwen3-8B-Base",
        "Qwen/Qwen3-8B-Base",
        "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        36,
        29,
        template_from="qwen3-8b",
    ),
    "llama-3.1-8b-base": LLMSettings(
        "Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B",
        "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        32,
        24,
        template_from="llama-3.1-8b-instruct",
    ),
    "olmo-3-7b-base": LLMSettings(
        "OLMo-3-7B",
        "allenai/Olmo-3-1025-7B",
        "a81bae42db3975be1671e27b9c9a56da1a9f980f",
        32,
        23,
        template_from="olmo-3-7b-instruct",
    ),
}
"""The six instruction-tuned LLMs and four base checkpoints. A base checkpoint is read at its
sibling's block, in its sibling's chat template."""

EMBEDDERS = {
    "qwen3-emb-0.6b": EmbedderSettings(
        "Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "last",
        None,
    ),
    "bge-large-en-v1.5": EmbedderSettings(
        "bge-large-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "cls",
        512,
    ),
    "gte-large-en-v1.5": EmbedderSettings(
        "gte-large-en-v1.5",
        "Alibaba-NLP/gte-large-en-v1.5",
        "104333d6af6f97649377c2afbde10a7704870c7b",
        "cls",
        512,
        code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4",
    ),
    "all-mpnet-base-v2": EmbedderSettings(
        "all-mpnet-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "mean",
        512,
    ),
    "e5-large-v2": EmbedderSettings(
        "e5-large-v2",
        "intfloat/e5-large-v2",
        "f169b11e22de13617baa190a028a32f3493550b6",
        "mean",
        512,
        prefix="query: ",
    ),
    "all-MiniLM-L6-v2": EmbedderSettings(
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "mean",
        512,
    ),
    "bge-small-en-v1.5": EmbedderSettings(
        "bge-small-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "cls",
        512,
    ),
    "bge-base-en-v1.5": EmbedderSettings(
        "bge-base-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "cls",
        512,
    ),
    "qwen3-emb-4b": EmbedderSettings(
        "Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-4B",
        "5cf2132abc99cad020ac570b19d031efec650f2b",
        "last",
        None,
    ),
    "qwen3-emb-8b": EmbedderSettings(
        "Qwen3-Embedding-8B",
        "Qwen/Qwen3-Embedding-8B",
        "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "last",
        None,
    ),
}
"""The ten embedders, used without an instruction. Qwen3-Embedding reads in its document mode:
the text alone, with the end token its tokenizer appends as the pooling token."""

INSTRUCTION_TUNED = [key for key, llm in LLMS.items() if llm.template_from is None]
BASE_CHECKPOINTS = [key for key, llm in LLMS.items() if llm.template_from is not None]
CORE_EMBEDDERS = [
    "qwen3-emb-0.6b",
    "bge-large-en-v1.5",
    "gte-large-en-v1.5",
    "all-mpnet-base-v2",
    "e5-large-v2",
]
"""The five embedders of the first grid."""
MORE_EMBEDDERS = [
    "all-MiniLM-L6-v2",
    "bge-small-en-v1.5",
    "bge-base-en-v1.5",
    "qwen3-emb-4b",
    "qwen3-emb-8b",
]
"""Five more embedders; with the core five, every LLM is paired with all ten."""
FAMILY_LLMS = ["qwen3-1.7b", "llama-3.1-8b-instruct", "olmo-3-7b-instruct"]
"""One LLM of each family, for the experiments that run on three LLMs."""
ITEM_SETS = ["fewrel", "fewnerd", "fewevent", "instructstsb"]
"""The datasets whose items each come with their own instruction."""

MAX_TEXT_TOKENS = 400
"""Every text is truncated to this many of the LLM's tokens; both models read the truncated text."""
RELAY = {"beta": 1.0, "bias": "mass_preserving", "layers": None}
"""The paper's relay for every pair; layers None means every layer."""

SOFTWARE = {
    "gpu": "torch 2.8.0, transformers 5.17.0, sentence-transformers 6.0.1 (Modal, H100)",
    "mac": "torch 2.14.0, transformers 5.17.0, sentence-transformers 6.0.1 "
    "(Apple silicon, MPS; the mean-pooling NYT and IE encodes)",
}
"""The software the paper's runs used."""

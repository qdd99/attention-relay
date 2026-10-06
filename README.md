# Attention Relay

![Paper](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b)
[![Website](https://img.shields.io/badge/Project-Page-blue)](https://qdd99.github.io/attention-relay)

This repository is for the paper "Lend Me Your Eyes: Instruction-Aware Text Embeddings via Attention Relay." It contains `attention_relay`, a library that makes an embedding model instruction-aware with no training, and the code to reproduce the paper's experiments.

## Setup

```bash
git clone https://github.com/qdd99/attention-relay && cd attention-relay
pip install .
```

Requires Python 3.10 or newer, PyTorch, and transformers 5.17 or newer. The larger LLMs need a GPU.

## Quick start

```python
from attention_relay import AttentionRelay

relay = AttentionRelay("Qwen/Qwen3-1.7B", "BAAI/bge-large-en-v1.5")

by_topic = relay.encode(articles, "What is the topic of news?")
by_place = relay.encode(articles, "Where did the news happen?")
released = relay.encode(articles)  # the embedding model as released
```

`encode` returns L2-normalized embeddings as a NumPy array, one row per text. The instruction can be
one string for every text or a list with one per text.

<details>
<summary><strong>Options</strong></summary>

| Argument | Default | Description |
|---|---|---|
| `llm` | required | an instruction-tuned LLM's repository id, or an `LLMReader` |
| `embedder` | required | an embedding model's repository id, or an `Embedder` |
| `block` | the paper's, else about 70% of the way up | the LLM block to read, counted from 0; for an LLM outside the Models table, see Other LLMs |
| `beta` | 1.0 | the relay's strength; 0 gives the embedding model as released |
| `bias` | `mass_preserving` | or `plain` |
| `layers` | all | for [CLS] and last-token pooling, the embedding model's layers to relay in |
| `device` | CUDA if available | where both models run; CPU otherwise |

`LLMReader(..., max_text_tokens=n)` truncates texts to their first n LLM tokens; by default they are
read whole. `LLMReader` also takes a `revision`, for the exact revisions in the Models tables, and a
`dtype`.

</details>

## Other LLMs

For an LLM outside the Models table, any block in its upper half works: about 70% of the way up is a safe default, and `AttentionRelay` uses it when no block is given:

```python
from attention_relay import AttentionRelay

relay = AttentionRelay("your/instruction-tuned-llm", "BAAI/bge-large-en-v1.5")
```

`select_block` finds the most question-sensitive one; give it texts and two questions about them:

```python
from attention_relay import AttentionRelay, LLMReader, select_block

reader = LLMReader("your/instruction-tuned-llm")
block = select_block(
    reader,
    (articles, ("What is the topic of news?", "Where did the news happen?")),
    (messages, ("How does the customer feel?", "What does the customer need?")),
)
relay = AttentionRelay(reader, "BAAI/bge-large-en-v1.5", block=block)
```

## Embedding models

Any Hugging Face embedding model with mean, [CLS] or last-token pooling works. `AttentionRelay` wraps it
in an `Embedder`, which reads the pooling from the model's sentence-transformers configuration. [CLS]
and last-token pooling need a model whose attention runs through transformers' attention interface,
as most current architectures' does. For settings a repository id can't carry, build the `Embedder`
yourself and pass it to `AttentionRelay`.

<details>
<summary><strong>Options</strong></summary>

| Argument | Default | Description |
|---|---|---|
| `model` | required | a repository id, a local directory or a loaded model |
| `tokenizer` | the model's | needed when `model` is a loaded model |
| `pooling` | the model's | `mean`, `cls` or `last` |
| `prefix` | none | a fixed string read before every text, such as e5's `query: `; it gets no relayed weight |
| `max_tokens` | the model's limit | texts are truncated to this many tokens |
| `revision` | latest | the model's revision |
| `dtype` | by pooling | float32 for mean pooling; for [CLS] and last-token pooling, bfloat16 on CUDA and float32 elsewhere |
| `device` | CUDA if available | CPU otherwise |
| `trust_remote_code`, `code_revision` | off | for models with custom code, such as gte-large-en-v1.5 |

</details>

<details>
<summary><strong>Notes</strong></summary>

- **e5-large-v2** expects every input to start with `query: `, as its model card prescribes for
  clustering and classification. Without the prefix it still runs, but not as intended. The prefix
  gets no relayed weight:

  ```python
  from attention_relay import AttentionRelay, Embedder

  embedder = Embedder("intfloat/e5-large-v2", prefix="query: ")
  relay = AttentionRelay("Qwen/Qwen3-1.7B", embedder)
  ```

- **gte-large-en-v1.5** runs its own modeling code, written for transformers 4, so it needs
  `trust_remote_code=True`. The library repairs that code for transformers 5 and relays through it;
  the paper pinned the code's revision:

  ```python
  embedder = Embedder(
      "Alibaba-NLP/gte-large-en-v1.5",
      trust_remote_code=True,
      code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4",
  )
  ```

</details>

## Reproducing the paper

```bash
uv sync --group reproduce

# The full run, on a CUDA GPU
uv run python -m reproduce.run --out runs/paper

# A short trial, on a Mac
uv run python -m reproduce.run --out runs/trial --limit 16 --device mps

# The paper's figures and tables, from a run
uv run python -m reproduce.paper --results runs/paper/results --out runs/paper

# ... or from the paper's own results
uv run python -m reproduce.paper --results results/paper --out runs/figures
```

The run has seven stages, in order: reading, blocks, grid, items, analyses, inputs and figures. Each
stage keeps what is already stored under `--out`, so an interrupted run resumes where it stopped. The
scores go to `<out>/results/*.json`.

<details>
<summary><strong>Command options</strong></summary>

| Argument | Default | Description |
|---|---|---|
| `--out` | required | the run directory |
| `--device` | `cuda` | `cuda`, `mps` or `cpu` |
| `--limit` | all | read only the first texts of each dataset, for a trial |
| `--only` | all | comma-separated stages to run |
| `--llms`, `--embedders` | all | comma-separated keys from the Models tables |
| `--workers` | 0 | processes for the NYT bootstrap intervals |

</details>

<details>
<summary><strong>What to expect</strong></summary>

A rerun on other hardware should reproduce the paper's numbers to within about half a point, not to the last digit.

- The clustering scores (NYT, and the V-measure of FewRel, FewNerd and FewEvent) average ten seeded k-means runs. The seeds give the same random draws on every machine, but a last-bit difference in the embeddings (another GPU or numeric type) or in the arithmetic (another CPU) can send a run to a different local optimum. A score can move by up to about half a point.
- IE and InstructSTSB use no k-means and move only with the embeddings. Relaying in float32 on a Mac instead of bfloat16 on an H100 moved one IE gain by about half a point.
- The gains' bootstrap intervals already include k-means' variation.
- Where two blocks are close, the label-free rule may choose the neighbouring one on other hardware (OLMo-3-7B-Instruct's blocks 23 and 22). The stages read each LLM at the paper's block; the blocks stage reports the rule's choice.
- On Python 3.12 and newer, the reproduction pins NumPy 2.5.3 and scikit-learn 1.9.1, the versions it was checked with.
</details>

<details>
<summary><strong>Notes</strong></summary>

- The model weights at the pinned revisions take about 140 GB of disk (110 GB for the ten LLMs, 30 GB for the ten embedding models). The LLMs' attention and the embeddings take about 13 GB more.
- Llama 3.1 checkpoints are gated on Hugging Face.
- We originally ran the paper's GPU passes on [Modal](https://modal.com/), and `reproduce/modal_run.py` does the same.

</details>

## Models

These are the paper's models, at the exact Hugging Face revisions every run used. `AttentionRelay`
reads each LLM at the block shown, counted from 0.

| Key | LLM | Block |
|---|---|---|
| `qwen3-0.6b` | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | 24 of 28 |
| `qwen3-1.7b` | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) | 24 of 28 |
| `qwen3-4b` | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) | 29 of 36 |
| `qwen3-8b` | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) | 29 of 36 |
| `llama-3.1-8b-instruct` | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | 24 of 32 |
| `olmo-3-7b-instruct` | [OLMo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc) | 23 of 32 |

| Key | Embedding model | Pooling |
|---|---|---|
| `qwen3-emb-0.6b` | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | last token |
| `qwen3-emb-4b` | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B/tree/5cf2132abc99cad020ac570b19d031efec650f2b) | last token |
| `qwen3-emb-8b` | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af) | last token |
| `bge-small-en-v1.5` | [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) | [CLS] |
| `bge-base-en-v1.5` | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5/tree/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a) | [CLS] |
| `bge-large-en-v1.5` | [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5/tree/d4aa6901d3a41ba39fb536a557fa166f842b0e09) | [CLS] |
| `gte-large-en-v1.5` | [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5/tree/104333d6af6f97649377c2afbde10a7704870c7b) | [CLS] |
| `all-MiniLM-L6-v2` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) | mean |
| `all-mpnet-base-v2` | [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/e8c3b32edf5434bc2275fc9bab85f82640a19130) | mean |
| `e5-large-v2` | [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2/tree/f169b11e22de13617baa190a028a32f3493550b6) | mean |

The reproduction also reads four base checkpoints (Qwen3-1.7B-Base, Qwen3-8B-Base, Llama-3.1-8B and
OLMo-3-7B), each at its instruction-tuned sibling's block and in its sibling's chat template;
[reproduce/settings.py](reproduce/settings.py) records their revisions.

## Citation

To be added with the arXiv version.

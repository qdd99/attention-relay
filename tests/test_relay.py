"""Tests for the relay and the embedder's three poolings."""

import numpy as np
import pytest
import torch
import transformers

from attention_relay.relay import FLOOR, Embedder, Encoding, relay_row

# ---- the relay's arithmetic ----------------------------------------------------------------------


def random_rows(seed=0, heads=3, positions=9):
    """Attention logits and rows, a text mask that leaves out the first and last positions, and
    positive carried weights that sum to one on the text."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(heads, positions, generator=generator) * 2
    text_mask = torch.zeros(positions, dtype=torch.bool)
    text_mask[1:-1] = True
    weights = torch.rand(positions, generator=generator) + 0.05
    weights = torch.where(text_mask, weights, torch.zeros_like(weights))
    return logits, logits.softmax(dim=-1), text_mask, weights / weights.sum()


@pytest.mark.parametrize("bias", ["mass_preserving", "plain"])
def test_beta_zero_returns_the_row(bias):
    _, attention, text_mask, weights = random_rows()
    relayed = relay_row(attention, text_mask, weights, beta=0.0, bias=bias)
    torch.testing.assert_close(relayed, attention)


@pytest.mark.parametrize("bias", ["mass_preserving", "plain"])
def test_uniform_weights_change_nothing(bias):
    _, attention, text_mask, _ = random_rows()
    uniform = text_mask.float() / text_mask.sum()
    torch.testing.assert_close(relay_row(attention, text_mask, uniform, bias=bias), attention)


@pytest.mark.parametrize("beta", [1.0, 2.0])
def test_mass_preserving_keeps_each_heads_attention_on_the_text(beta):
    _, attention, text_mask, weights = random_rows()
    relayed = relay_row(attention, text_mask, weights, beta=beta)
    torch.testing.assert_close(relayed[:, text_mask].sum(-1), attention[:, text_mask].sum(-1))
    torch.testing.assert_close(relayed[:, ~text_mask], attention[:, ~text_mask])
    share = attention[:, text_mask] * weights[text_mask] ** beta  # r' is proportional to r w^beta
    share = share / share.sum(-1, keepdim=True)
    relayed_share = relayed[:, text_mask] / relayed[:, text_mask].sum(-1, keepdim=True)
    torch.testing.assert_close(relayed_share, share)


@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_the_logit_form_equals_the_multiply_form(beta):
    logits, attention, text_mask, weights = random_rows()
    mass = attention[:, text_mask].sum(-1, keepdim=True)
    lifted = (attention[:, text_mask] * weights[text_mask] ** beta).sum(-1, keepdim=True)
    constant = torch.log(mass / lifted)  # the constant that keeps each head's mass on the text
    bias = torch.zeros_like(logits)
    bias[:, text_mask] = beta * torch.log(weights[text_mask]) + constant
    torch.testing.assert_close(
        (logits + bias).softmax(-1), relay_row(attention, text_mask, weights, beta)
    )


@pytest.mark.parametrize("beta", [1.0, 2.0])
def test_the_plain_bias_is_a_logit_bias_that_scales_the_odds_by_kappa(beta):
    logits, attention, text_mask, weights = random_rows()
    relayed = relay_row(attention, text_mask, weights, beta, bias="plain")

    n_text = text_mask.sum()
    bias = torch.zeros_like(logits)
    bias[:, text_mask] = beta * torch.log(n_text * weights[text_mask])  # c = beta log |T|
    torch.testing.assert_close((logits + bias).softmax(-1), relayed)

    mass, relayed_mass = attention[:, text_mask].sum(-1), relayed[:, text_mask].sum(-1)
    share = attention[:, text_mask] / mass[:, None]
    # the factor on each head's odds of attending to the text
    kappa = n_text**beta * (share * weights[text_mask] ** beta).sum(-1)
    odds = mass / (1 - mass)
    torch.testing.assert_close(relayed_mass / (1 - relayed_mass), kappa * odds)


def test_a_token_the_llm_ignores_keeps_almost_no_attention():
    _, attention, text_mask, weights = random_rows()
    weights[3] = 0.0
    relayed = relay_row(attention, text_mask, weights)
    assert relayed[:, 3].max() < 10 * FLOOR


def test_an_unknown_bias_is_an_error():
    _, attention, text_mask, weights = random_rows()
    with pytest.raises(ValueError, match="bias must be one of"):
        relay_row(attention, text_mask, weights, bias="other")


# ---- the relay inside small random-weight models -------------------------------------------------

ENCODER = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3}
ENCODER |= {"num_attention_heads": 4, "vocab_size": 500, "pad_token_id": 0}
DECODER = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3, "head_dim": 16}
DECODER |= {"num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 500}
DECODER |= {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}

ARCHITECTURES = {  # name: (make a config, pooling)
    "BERT": (lambda: transformers.BertConfig(**ENCODER), "cls"),
    "XLM-R": (lambda: transformers.XLMRobertaConfig(**ENCODER), "cls"),
    "ModernBERT (local attention)": (
        lambda: transformers.ModernBertConfig(
            **ENCODER, global_attn_every_n_layers=2, local_attention=4
        ),
        "cls",
    ),
    "DistilBERT (no layer index)": (
        lambda: transformers.DistilBertConfig(
            dim=64, hidden_dim=128, n_layers=3, n_heads=4, vocab_size=500, pad_token_id=0
        ),
        "cls",
    ),
    "Qwen3 (causal, last token)": (lambda: transformers.Qwen3Config(**DECODER), "last"),
}


def random_encodings(seed=0):
    """Texts of different lengths; the first and last tokens are outside the text."""
    generator = np.random.default_rng(seed)
    encodings = []
    for length in (7, 12, 9, 15):
        ids = generator.integers(3, 500, length).tolist()
        text_mask = [0 < position < length - 1 for position in range(length)]
        encodings.append(Encoding(ids, [(0, 0)] * length, [not t for t in text_mask], text_mask))
    return encodings


def random_weights(encodings, seed=1):
    generator = np.random.default_rng(seed)
    weights = []
    for encoding in encodings:
        values = generator.dirichlet(np.full(len(encoding.ids), 0.3))
        values = np.where(encoding.text_mask, values, 0.0)
        weights.append(values / values.sum())
    return weights


def build_embedder(architecture):
    make_config, pooling = ARCHITECTURES[architecture]
    torch.manual_seed(0)
    model = transformers.AutoModel.from_config(make_config())
    reference = transformers.AutoModel.from_config(make_config(), attn_implementation="eager")
    reference.load_state_dict(model.state_dict())
    return Embedder(model, pooling=pooling, device="cpu"), reference.eval()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_pooling_rows_match_eager_attention(architecture):
    embedder, reference = build_embedder(architecture)
    encodings = random_encodings()
    rows = embedder.pooling_attention(encodings, max_batch_tokens=30)  # batches with padding
    for encoding, text_rows in zip(encodings, rows, strict=True):
        with torch.no_grad():
            outputs = reference(input_ids=torch.tensor([encoding.ids]), output_attentions=True)
        position = 0 if embedder.pooling == "cls" else len(encoding.ids) - 1
        expected = torch.stack([layer[0, :, position, :] for layer in outputs.attentions])
        np.testing.assert_allclose(text_rows, expected.numpy(), atol=1e-5)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_relays_that_should_change_nothing_change_nothing(architecture):
    embedder, _ = build_embedder(architecture)
    encodings = random_encodings()
    released = embedder.embed(encodings)
    uniform = [np.array(e.text_mask, float) / sum(e.text_mask) for e in encodings]
    unchanged = {
        "beta = 0": embedder.embed(encodings, random_weights(encodings), beta=0.0),
        "uniform weights": embedder.embed(encodings, uniform),
        "uniform weights, plain bias": embedder.embed(encodings, uniform, bias="plain"),
        "no layers": embedder.embed(encodings, random_weights(encodings), layers=[]),
    }
    for name, embeddings in unchanged.items():
        np.testing.assert_allclose(embeddings, released, atol=1e-5, err_msg=name)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_relaying_the_weights_changes_the_embedding(architecture):
    embedder, _ = build_embedder(architecture)
    encodings = random_encodings()
    released = embedder.embed(encodings)
    relayed = embedder.embed(encodings, random_weights(encodings))
    first_layer_only = embedder.embed(encodings, random_weights(encodings), layers=[0])
    # far above float32 noise (1e-6); random-weight models move little, so no larger bar
    assert np.linalg.norm(relayed - released, axis=1).min() > 1e-3
    assert np.linalg.norm(relayed - first_layer_only, axis=1).min() > 1e-3


def test_in_a_causal_embedder_only_the_pooling_token_changes():
    embedder, _ = build_embedder("Qwen3 (causal, last token)")
    encodings = random_encodings()
    batch = list(range(len(encodings)))
    (_, ids, attention_mask) = next(embedder._batches(encodings, max_batch_tokens=10**6))
    relay = embedder._relay(
        batch, encodings, attention_mask, random_weights(encodings), 1.0, "mass_preserving", None
    )
    released = embedder._forward(ids, attention_mask, None)
    relayed = embedder._forward(ids, attention_mask, relay)
    for row, encoding in enumerate(encodings):
        last = len(encoding.ids) - 1
        torch.testing.assert_close(relayed[row, :last], released[row, :last], rtol=0, atol=0)
        assert not torch.allclose(relayed[row, last], released[row, last])


def test_mean_pooling_averages_with_the_relayed_weights():
    torch.manual_seed(0)
    model = transformers.AutoModel.from_config(transformers.BertConfig(**ENCODER))
    embedder = Embedder(model, pooling="mean", device="cpu")
    encodings = random_encodings()
    weights = random_weights(encodings)
    states = embedder.token_states(encodings)

    def normalized(vectors):
        vectors = np.stack(vectors)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    np.testing.assert_allclose(
        embedder.embed(encodings), normalized([s.mean(axis=0) for s in states]), atol=1e-6
    )
    np.testing.assert_allclose(
        embedder.embed(encodings, weights),
        normalized([w @ s for w, s in zip(weights, states, strict=True)]),
        atol=1e-6,
    )
    squared = [w**2 / (w**2).sum() for w in weights]
    np.testing.assert_allclose(
        embedder.embed(encodings, weights, beta=2.0),
        normalized([w @ s for w, s in zip(squared, states, strict=True)]),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="layer range"):
        embedder.embed(encodings, weights, layers=[0])


# ---- released models, when they are in the local cache -------------------------------------------

RELEASED = {
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
}
TEXTS = [
    "It's really confusing, how exactly are you determining your exchange rates?",
    "WASHINGTON - The Senate voted on Tuesday to confirm the nominee, ending a long standoff.",
    "Great service but the app crashed twice!",
]


@pytest.mark.parametrize("repository", RELEASED)
def test_the_released_embedding_matches_sentence_transformers(repository):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    try:
        embedder = Embedder(repository, revision=RELEASED[repository], device="cpu")
        reference = sentence_transformers.SentenceTransformer(
            repository, revision=RELEASED[repository], device="cpu", local_files_only=True
        )
    except OSError:
        pytest.skip(f"{repository} is not in the local cache")
    ours = embedder.embed(embedder.tokenize(TEXTS))
    theirs = reference.encode(TEXTS, normalize_embeddings=True, convert_to_numpy=True)
    assert (ours * theirs).sum(axis=1).min() > 0.99999

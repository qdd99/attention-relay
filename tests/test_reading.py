"""Tests for the reading weights.

The reference is transformers' own eager attention: its attention weights at the last position,
summed over heads on the text tokens and renormalized. The models are small, randomly initialized
versions of each architecture, so the tests download nothing.
"""

import numpy as np
import pytest
import torch
import transformers

from attention_relay.reading import (
    LLMReader,
    build_prompt,
    read_sequences,
    truncate_text,
    use_reading_attention,
)

SMALL = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 500,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}
WINDOW = 6  # shorter than the test sequences, so sliding windows bind
LAYER_TYPES = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
NO_HEAD_DIM = {key: value for key, value in SMALL.items() if key != "head_dim"}

ARCHITECTURES = {
    "Qwen3": lambda: transformers.Qwen3Config(**SMALL),
    "Llama": lambda: transformers.LlamaConfig(**SMALL),
    "OLMo 3 (sliding windows)": lambda: transformers.Olmo3Config(
        **SMALL, sliding_window=WINDOW, layer_types=LAYER_TYPES
    ),
    "Gemma 2 (logit soft-capping)": lambda: transformers.Gemma2Config(
        **SMALL, attn_logit_softcapping=5.0, sliding_window=WINDOW
    ),
    "Gemma 3": lambda: transformers.Gemma3TextConfig(**SMALL, sliding_window=WINDOW),
    "Phi-3 (fused projections)": lambda: transformers.Phi3Config(**NO_HEAD_DIM),
    "gpt-oss (attention sinks)": lambda: transformers.GptOssConfig(
        **SMALL, num_local_experts=4, num_experts_per_tok=2, sliding_window=WINDOW
    ),
}


def random_sequences(seed: int = 0) -> list[tuple[list[int], list[bool]]]:
    """Token sequences of different lengths, each with a text span ending before the last token."""
    generator = np.random.default_rng(seed)
    sequences = []
    for length in (9, 17, 12, 14, 10):
        ids = generator.integers(3, SMALL["vocab_size"], length).tolist()
        start = int(generator.integers(1, 4))
        end = length - int(generator.integers(2, 4))
        sequences.append((ids, [start <= position < end for position in range(length)]))
    return sequences


def build_models(make_config):
    """The same random weights twice: with the recording attention and with eager attention."""
    torch.manual_seed(0)
    model = transformers.AutoModel.from_config(make_config()).eval()
    for module in model.modules():  # attention sinks start at zero; make them matter
        if hasattr(module, "sinks"):
            torch.nn.init.normal_(module.sinks)
    # a config of its own: switching one model's attention must not switch the other's
    reference = transformers.AutoModel.from_config(make_config(), attn_implementation="eager")
    reference.load_state_dict(model.state_dict())
    reference.eval()
    use_reading_attention(model)
    return model, reference


def eager_reading_weights(reference, ids, text_mask) -> np.ndarray:
    """The reference: eager attention weights of the last position, one unpadded sequence."""
    with torch.no_grad():
        outputs = reference(input_ids=torch.tensor([ids]), output_attentions=True, use_cache=False)
    mask = torch.tensor(text_mask)
    rows = []
    for attention in outputs.attentions:  # one [1, heads, length, length] tensor per block
        on_text = attention[0, :, -1, :].sum(dim=0)[mask].double()
        rows.append(on_text / on_text.sum())
    return torch.stack(rows).numpy()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_matches_eager_attention(architecture):
    model, reference = build_models(ARCHITECTURES[architecture])
    sequences = random_sequences()
    blocks = range(model.config.num_hidden_layers)
    # three sequences share each batch, so most of them are left-padded
    weights = read_sequences(model, sequences, blocks, pad_token_id=0, max_batch_size=3)
    for (ids, text_mask), sequence_weights in zip(sequences, weights, strict=True):
        expected = eager_reading_weights(reference, ids, text_mask)
        np.testing.assert_allclose(sequence_weights, expected, atol=1e-5)


def test_stopping_after_an_early_block_changes_nothing():
    model, _ = build_models(ARCHITECTURES["Qwen3"])
    sequences = random_sequences()
    every_block = read_sequences(model, sequences, [0, 1, 2, 3], pad_token_id=0)
    block_one = read_sequences(model, sequences, [1], pad_token_id=0)
    for all_rows, one_row in zip(every_block, block_one, strict=True):
        np.testing.assert_array_equal(all_rows[1:2], one_row)


def test_rows_sum_to_one():
    model, _ = build_models(ARCHITECTURES["Llama"])
    for weights in read_sequences(model, random_sequences(), [0, 3], pad_token_id=0):
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_a_block_outside_the_model_is_an_error():
    model, _ = build_models(ARCHITECTURES["Qwen3"])
    with pytest.raises(ValueError, match="blocks must lie in"):
        read_sequences(model, random_sequences(), [4], pad_token_id=0)


def test_a_model_without_the_recording_attention_is_an_error():
    model = transformers.AutoModel.from_config(transformers.Qwen3Config(**SMALL)).eval()
    with pytest.raises(ValueError, match="use_reading_attention"):
        read_sequences(model, random_sequences(), [0], pad_token_id=0)


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except OSError:
        pytest.skip("the Qwen3 tokenizer is not available")


def test_the_prompt_marks_exactly_the_text(qwen3_tokenizer):
    text = "It's really confusing, how exactly are you determining your exchange rates?"
    instruction = "How does the customer feel?"
    prompt = build_prompt(qwen3_tokenizer, text, instruction)

    ids = np.array(prompt.ids)
    text_ids = qwen3_tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids[np.array(prompt.text_mask)].tolist() == text_ids

    message = f"Text:\n{text}\n\n{instruction}\n\nAnswer in one word."
    rendered = qwen3_tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert qwen3_tokenizer.decode(prompt.ids) == rendered
    assert rendered.endswith("<think>\n\n</think>\n\n")  # thinking is off; the answer comes next


def test_truncation_keeps_the_first_tokens(qwen3_tokenizer):
    text = " ".join(f"word{index}" for index in range(300))
    truncated = truncate_text(qwen3_tokenizer, text, 50)
    assert len(qwen3_tokenizer(truncated, add_special_tokens=False)["input_ids"]) == 50
    assert text.startswith(truncated)
    assert truncate_text(qwen3_tokenizer, text, None) == text


def test_the_reader_returns_offsets_into_the_text_it_read(qwen3_tokenizer):
    torch.manual_seed(0)
    config = transformers.Qwen3Config(**{**SMALL, "vocab_size": len(qwen3_tokenizer)})
    model = transformers.AutoModel.from_config(config)
    reader = LLMReader(model, qwen3_tokenizer, device="cpu", max_text_tokens=8)
    text = "Great service but the app crashed twice, and nobody answered my emails."
    (reading,) = reader.read([text], "What does the customer need?", blocks=[1, 3])

    assert text.startswith(reading.text)
    assert reading.blocks == (1, 3)
    assert reading.weights.shape == (2, len(reading.offsets)) == (2, 8)
    assert reading.offsets[0][0] == 0 and reading.offsets[-1][1] == len(reading.text)
    np.testing.assert_allclose(reading.at(3).sum(), 1.0)


def test_unnormalized_rows_hold_the_head_summed_attention_on_the_text():
    model, _ = build_models(ARCHITECTURES["Qwen3"])
    sequences = random_sequences()
    normalized = read_sequences(model, sequences, [0, 3], pad_token_id=0)
    raw = read_sequences(model, sequences, [0, 3], pad_token_id=0, normalize=False)
    heads = model.config.num_attention_heads
    for weights, sums in zip(normalized, raw, strict=True):
        assert ((sums.sum(axis=1) > 0) & (sums.sum(axis=1) <= heads)).all()
        np.testing.assert_allclose(sums / sums.sum(axis=1, keepdims=True), weights)

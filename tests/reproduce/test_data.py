"""Tests for the reproduction's datasets and settings."""

import math

import pytest

from attention_relay import KNOWN_BLOCKS
from attention_relay.relay import POOLINGS
from reproduce import data, settings


def load_or_skip(loader, *args):
    try:
        return loader(*args)
    except Exception as error:  # offline without a cached copy, or no datasets library
        pytest.skip(f"the dataset is not available: {type(error).__name__}")


def test_nyt_has_2144_articles_with_both_labels():
    nyt = load_or_skip(data.load_nyt)
    assert len(nyt.texts) == 2144 == len(set(nyt.texts))
    assert all(len(labels) == 2144 for labels in nyt.labels.values())
    assert set(nyt.labels) == set(data.QUESTIONS["nyt"])


def test_intent_emotion_scores_2038_triplets_per_question():
    ie = load_or_skip(data.load_intent_emotion)
    assert len({anchor for anchor, _, _ in ie.triplets["intent"]}) == data.IE_ANCHORS
    assert {aspect: len(rows) for aspect, rows in ie.triplets.items()} == {
        "emotion": 2038,
        "intent": 2038,
    }
    texts = set(ie.texts)
    assert all(text in texts for rows in ie.triplets.values() for row in rows for text in row)


@pytest.mark.parametrize("name", ["fewrel", "fewnerd", "fewevent"])
def test_clustering_sets_come_with_their_instruction(name):
    dataset = load_or_skip(data.load_clustering_set, name)
    assert len(dataset.texts) == len(dataset.labels) > 0
    assert dataset.instruction == data.INSTRUCTIONS[name]


def test_instruct_stsb_has_2758_pairs_of_distinct_items():
    stsb = load_or_skip(data.load_instruct_stsb)
    assert len(stsb.pairs) == 2758
    assert len(set(stsb.items)) == len(stsb.items)
    assert all(stsb.items[a][1] == stsb.items[b][1] for a, b, _ in stsb.pairs)  # one instruction


def test_every_block_lies_where_the_label_free_rule_looks():
    for llm in settings.LLMS.values():
        assert math.ceil(llm.n_blocks / 2) <= llm.block <= llm.n_blocks - 2


def test_base_checkpoints_follow_their_instruction_tuned_sibling():
    for key in settings.BASE_CHECKPOINTS:
        base = settings.LLMS[key]
        sibling = settings.LLMS[base.template_from]
        assert base.template_from in settings.INSTRUCTION_TUNED
        assert (base.block, base.n_blocks) == (sibling.block, sibling.n_blocks)


def test_the_library_reads_the_paper_llms_at_the_same_blocks():
    from_settings = {
        settings.LLMS[key].repository: settings.LLMS[key].block
        for key in settings.INSTRUCTION_TUNED
    }
    assert from_settings == KNOWN_BLOCKS


def test_every_llm_meets_all_ten_embedders():
    assert sorted(settings.CORE_EMBEDDERS + settings.MORE_EMBEDDERS) == sorted(settings.EMBEDDERS)
    assert set(settings.FAMILY_LLMS) <= set(settings.INSTRUCTION_TUNED)
    assert all(embedder.pooling in POOLINGS for embedder in settings.EMBEDDERS.values())

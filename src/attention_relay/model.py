"""Attention Relay: an LLM's reading weights, relayed into an embedder's pooling."""

from collections.abc import Collection, Sequence

import numpy as np
import torch

from attention_relay.align import alignment_matrix, carry_weights
from attention_relay.blocks import KNOWN_BLOCKS, default_block
from attention_relay.reading import LLMReader, Reading, truncate_text
from attention_relay.relay import Embedder, Encoding


class AttentionRelay:
    """Instruction-aware text embeddings from an instruction-agnostic embedder, without training.

    The LLM reads each text under the instruction; its reading weights at one block are carried
    through characters to the embedder's tokens and relayed into the embedder's pooling. The
    embedder never sees the instruction.

    Example:
        relay = AttentionRelay("Qwen/Qwen3-1.7B", "BAAI/bge-large-en-v1.5")
        by_place = relay.encode(articles, "Where did the news happen?")
        released = relay.encode(articles)
    """

    def __init__(
        self,
        llm: str | LLMReader,
        embedder: str | Embedder,
        *,
        block: int | None = None,
        beta: float = 1.0,
        bias: str = "mass_preserving",
        layers: Collection[int] | None = None,
        device: str | torch.device | None = None,
    ):
        """Load both models.

        Args:
            llm: An instruction-tuned LLM's repository id, or an `LLMReader` for other settings.
            embedder: An embedder's repository id, or an `Embedder` for other settings, such as
                e5's "query: " prefix.
            block: The LLM's block to read, counted from 0. Defaults to its block in
                `KNOWN_BLOCKS`, and to about 70% of the way up for any other LLM
                (`default_block`); `select_block` finds the most question-sensitive one.
            beta: The relay's strength; 0 gives the embedder as released.
            bias: "mass_preserving" (the default) or "plain".
            layers: For CLS and last-token pooling, the embedder layers the relay acts in; None
                means all of them.
            device: Where both models run; defaults to CUDA when available, otherwise the CPU.
        """
        self.reader = llm if isinstance(llm, LLMReader) else LLMReader(llm, device=device)
        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder(embedder, device=device)
        if block is None:
            block = KNOWN_BLOCKS.get(self.reader.name)
            if block is None:
                block = default_block(self.reader.n_blocks)
        self.block = block
        self.beta = beta
        self.bias = bias
        self.layers = layers

    def read(self, texts: Sequence[str], instruction: str | Sequence[str]) -> list[Reading]:
        """Where the LLM looks in each text under the instruction, at the relay's block."""
        return self.reader.read(texts, instruction, blocks=self.block)

    def encode(
        self, texts: Sequence[str], instruction: str | Sequence[str] | None = None
    ) -> np.ndarray:
        """Embed texts under an instruction.

        Args:
            texts: The texts. Both the relayed and the released embedding read each text as the
                LLM reads it, truncated to the reader's `max_text_tokens`.
            instruction: One instruction for every text, or one per text. None gives the
                embedder as released.

        Returns:
            L2-normalized embeddings of shape [len(texts), dimension].
        """
        if instruction is None:
            read_texts = [
                truncate_text(self.reader.tokenizer, text, self.reader.max_text_tokens)
                for text in texts
            ]
            return self.embedder.embed(self.embedder.tokenize(read_texts))

        readings = self.read(texts, instruction)
        encodings = self.embedder.tokenize([reading.text for reading in readings])
        weights = [
            self.carry(reading, encoding)
            for reading, encoding in zip(readings, encodings, strict=True)
        ]
        return self.embedder.embed(
            encodings, weights, beta=self.beta, bias=self.bias, layers=self.layers
        )

    def carry(self, reading: Reading, encoding: Encoding) -> np.ndarray:
        """The LLM's reading weights on the embedder's tokens of the same text."""
        matrix = alignment_matrix(
            reading.text,
            reading.offsets,
            encoding.offsets,
            encoding.special,
            len(self.embedder.prefix),
        )
        return carry_weights(matrix, reading.at(self.block), np.array(encoding.text_mask))

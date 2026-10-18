# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in INT8 vocabulary heads; original logits processor owns semantics."""

import logging
import os

import torch
from torch import nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod,
)

from .w8_dense import PackedW8
from .w8_dense_policy import HEAD_SCOPES, HEAD_SHAPE, head_segments

log = logging.getLogger("vllm.qwen38.w8_head")


class PackedW8Head(nn.Module):
    """Store all vocabulary rows once; select range segments without unpacking.

    Full logits (including prefill) retain the full vocabulary and BF16 dtype.
    A reduced draft evaluates only its configured ranges. N padding is private
    to Marlin and trimmed BEFORE any soft cap, scale, max, or token mapping.
    """

    def __init__(self, weight, group_size=-1, ranges=()):
        super().__init__()
        if getattr(weight, "_vllm_is_uva_offloaded", False):
            raise ValueError("W8 head requires resident weights")
        self.n, self.k = weight.shape
        self.ranges = tuple(ranges)
        self.segments = head_segments(self.n, self.ranges)
        self.parts = nn.ModuleList()
        for start, end in self.segments:
            n = end - start
            padded_n = ((n + 63) // 64) * 64
            selected = weight[start:end]
            if n != padded_n:
                selected = torch.nn.functional.pad(selected, (0, 0, 0, padded_n - n))
            self.parts.append(PackedW8(selected, "int8", group_size))

    @property
    def storage_bytes(self):
        return sum(part.storage_bytes for part in self.parts)

    def project_range(self, x, start, end):
        # Profile/capture may request the full vocabulary or another subrange.
        # Reuse overlapping packed segments and slice their logits: no repack,
        # device reads, host sync, or persistent mutation during graph capture.
        if not 0 <= start < end <= self.n:
            raise ValueError("W8 head range must be nonempty and in bounds")
        outputs = []
        for (a, b), part in zip(self.segments, self.parts):
            lo, hi = max(a, start), min(b, end)
            if lo < hi:
                outputs.append(part(x)[..., lo - a:hi - a])
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)

    def forward(self, x, bias=None):
        outputs = [part(x)[..., :b - a]
                   for (a, b), part in zip(self.segments, self.parts)]
        result = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return result if bias is None else result + bias


class Qwen38W8HeadMethod(QuantizeMethodBase):
    # Enrollment follows construction, but keep the original loader contract.
    create_weights = UnquantizedEmbeddingMethod.create_weights

    def __init__(self, config, name, ranges=()):
        self.config, self.name, self.ranges = config, name, ranges
        # The serving loader replaces MTP's head with the target head AFTER
        # both models finalize. A single selected caller must keep the original
        # weight for the other caller. Sharing adds no second BF16 allocation.
        self.preserve_bf16 = name == "head" and not HEAD_SCOPES <= config.scopes
        self.bf16_method = UnquantizedEmbeddingMethod()

    def process_weights_after_loading(self, layer):
        if hasattr(layer, "_qwen38_w8_head"):
            return
        weight = layer.weight
        before = weight.numel() * weight.element_size()
        packed = PackedW8Head(weight, self.config.group_size, self.ranges)
        layer.add_module("_qwen38_w8_head", packed)
        if not self.preserve_bf16:
            del layer.weight
        del weight
        retained = before if self.preserve_bf16 else 0
        after = packed.storage_bytes + retained
        log.warning("W8_HEAD %s group=%d ranges=%s bf16_bytes=%d w8_bytes=%d "
                    "retained_bf16_bytes=%d "
                    "freed_bytes=%d freed_GiB=%.6f (live tensors, not allocator reserve)",
                    self.name, self.config.group_size, self.ranges, before,
                    packed.storage_bytes, retained, before - after,
                    (before - after) / 1024**3)

    def apply(self, layer, x, bias=None):
        if self.name == "head" and "head" not in self.config.scopes:
            return self.bf16_method.apply(layer, x, bias)
        return layer._qwen38_w8_head(x, bias)

    def draft_packed(self, layer):
        return layer._qwen38_w8_head if "draft_head" in self.config.scopes else None

    def apply_draft(self, layer, x, bias=None):
        packed = self.draft_packed(layer)
        if packed is None:
            return self.bf16_method.apply(layer, x, bias)
        return packed(x, bias)


class DraftLogitsProcessor(LogitsProcessor):
    """Caller-specific full/prefill/greedy dispatch, even on a shared head."""

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        method = lm_head.quant_method
        if isinstance(method, Qwen38W8HeadMethod):
            if self.head_dtype is not None and self.head_dtype != hidden_states.dtype:
                raise ValueError("W8 shared heads require BF16 or unspecified draft head_dtype")
            return method.apply_draft(lm_head, hidden_states, embedding_bias)
        return super()._apply_head(lm_head, hidden_states, embedding_bias)


def enroll_heads(module, dtype, config):
    is_draft = type(module).__name__ == "Qwen3_8FlashNextMTP"
    scope = "draft_head" if is_draft else "head"
    # Prepare the target for a draft-only scope too: it becomes the shared
    # head after loading. Its target apply() remains on the original BF16 path.
    if (is_draft and scope not in config.scopes) or not config.scopes & HEAD_SCOPES:
        return 0
    head = getattr(module, "lm_head", None)
    # PPMissingLayer has no weight: a head scope is stage-1-only under PP2.
    if head is None or not hasattr(head, "weight"):
        return 0
    if (not isinstance(head, ParallelLMHead) or head.tp_size != 1
            or tuple(head.weight.shape) != HEAD_SHAPE):
        raise ValueError("W8 heads require an untied TP1 (248320,2560) ParallelLMHead")
    if getattr(getattr(module, "config", None), "tie_word_embeddings", False):
        raise ValueError("W8 heads do not support tied embeddings")
    if (dtype != torch.bfloat16 or head.weight.dtype != torch.bfloat16
            or module.logits_processor.head_dtype not in (None, torch.bfloat16)):
        raise ValueError("W8 heads require BF16 activations, weights and head_dtype")
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("W8 heads require SM80")
    if type(head.quant_method) is not UnquantizedEmbeddingMethod:
        raise ValueError("W8 head is already quantized or wrapped (LoRA unsupported)")
    ranges = (getattr(module, "_draft_vocab_ranges", None) or ()) if is_draft else ()
    if not is_draft and "draft_head" in config.scopes:
        value = os.environ.get("VLLM_MTP_DRAFT_VOCAB_RANGES", "")
        if value:
            from .draft_vocab import parse_ranges
            ranges = parse_ranges(value, HEAD_SHAPE[0])
    # Ranges came from parse_ranges in the MTP constructor. Without the draft
    # restriction this is a single full-vocabulary W8 head, same as target.
    head.quant_method = Qwen38W8HeadMethod(config, scope, ranges)
    log.warning("W8_HEAD enrolled %s: INT8 changes selected caller numerics, "
                "group=%d ranges=%s target_w8=%s draft_w8=%s retain_bf16=%s",
                scope, config.group_size, ranges, "head" in config.scopes,
                "draft_head" in config.scopes, head.quant_method.preserve_bf16)
    return 1

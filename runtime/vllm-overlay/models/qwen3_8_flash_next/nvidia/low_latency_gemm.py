# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next decode GEMM selection on Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

QWEN38NEXT_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
    },
}


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen38NextLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen3_8_flash_next_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen38NextLowLatencyLinearMethod(
    _Qwen38NextLowLatencyApply, UnquantizedLinearMethod
):
    pass


class Qwen38NextLowLatencyEmbeddingMethod(
    _Qwen38NextLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


import os as _os
_TRITON_SKINNY = _os.environ.get("QWEN38_TRITON_SKINNY", "0") == "1"
_GLUON_THIN = _os.environ.get("QWEN38_GLUON_THIN", "0") in ("1", "exact")
_HORIZONTAL = _os.environ.get("QWEN38_SKINNY_HORIZONTAL", "0") == "1"
_SKINNY_MAX_M = 16 if _os.environ.get("QWEN38_SKINNY_M16", "0") == "1" else 8


def _qwen38next_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _TRITON_SKINNY:
        if _GLUON_THIN and _runtime_ok(x, weight):
            from .gluon_thin import try_mm
            result = try_mm(x, weight)
            if result is not None:
                return result
        if x.dim() == 2 and 1 <= x.shape[0] <= _SKINNY_MAX_M and _runtime_ok(x, weight):
            from .triton_skinny import skinny_mm
            return skinny_mm(x, weight)
        return torch.nn.functional.linear(x, weight)
    plan = QWEN38NEXT_GEMM_PLANS.get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


def _qwen38next_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen3_8_flash_next_low_latency_gemm",
    op_func=_qwen38next_low_latency_gemm,
    fake_impl=_qwen38next_low_latency_gemm_fake,
)


def enable_qwen38next_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    # Enroll before skinny methods are selected and before checkpoint loading.
    # The loader finalizes each selected BF16 matrix into Marlin exactly once.
    if _os.environ.get("QWEN38_W8_DENSE", "").strip().lower() not in ("", "0", "off"):
        from .w8_dense import enroll
        enroll(module, dtype)
    import logging
    log = logging.getLogger("vllm.qwen38.triton_skinny")
    # SM 8.6 (RTX 3090) and SM 8.9 (RTX 4090, issue #40: same kernel, identical output, ~10% faster decode).
    supported = any(current_platform.is_device_capability(cc) for cc in ((8, 0), (8, 6), (8, 9)))   # + SM 8.0 (GA100, CMP 170HX)
    if _TRITON_SKINNY and not supported:
        log.warning("QWEN38_TRITON_SKINNY=1 ignored: the skinny GEMM is enabled only on SM 8.0, 8.6 and 8.9 GPUs")
    if _TRITON_SKINNY and dtype == torch.bfloat16 and supported:
        shapes = []
        for child in module.modules():
            method = getattr(child, "quant_method", None)
            if isinstance(child, ParallelLMHead) and getattr(method, "preserve_bf16", False):
                # W8 shared heads retain BF16 for the unselected caller. Keep
                # that caller's original skinny/full-prefill dispatch as well.
                method.bf16_method = Qwen38NextLowLatencyEmbeddingMethod()
            is_linear = (isinstance(child, LinearBase)
                         and type(child.quant_method) is UnquantizedLinearMethod)
            is_head = (isinstance(child, ParallelLMHead)
                       and type(child.quant_method) is UnquantizedEmbeddingMethod)
            weight = getattr(child, "weight", None)
            if not (is_linear or is_head) or weight is None or weight.dim() != 2:
                continue
            if weight.dtype != torch.bfloat16 or not weight.is_cuda:
                continue
            child.quant_method = (Qwen38NextLowLatencyLinearMethod() if is_linear
                                  else Qwen38NextLowLatencyEmbeddingMethod())
            shapes.append(tuple(weight.shape))
        from collections import Counter
        log.warning("Triton skinny GEMM enabled for %d modules: %s", len(shapes),
                    sorted(Counter(shapes).items()))
        return
    if dtype != torch.bfloat16 or not _is_sm103():
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = QWEN38NEXT_GEMM_PLANS.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen38NextLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen38NextLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)


def _horizontal_gemm(x: torch.Tensor, weight: torch.Tensor,
                     widths: list[int]) -> torch.Tensor:
    from .triton_skinny import horizontal_mm
    return horizontal_mm(x, weight, widths)


def _horizontal_gemm_fake(x: torch.Tensor, weight: torch.Tensor,
                          widths: list[int]) -> torch.Tensor:
    return x.new_empty((x.shape[0] * weight.shape[0],))


direct_register_custom_op(
    op_name="qwen38_horizontal_gemm", op_func=_horizontal_gemm,
    fake_impl=_horizontal_gemm_fake,
)


def horizontal_linear(owner: nn.Module, x: torch.Tensor):
    """Return contiguous output views, or None to retain standard dispatch."""
    weight = getattr(owner, "_qwen38_horizontal_weight", None)
    if (weight is None or x.dim() != 2 or not 1 <= x.shape[0] <= _SKINNY_MAX_M
            or not _runtime_ok(x, weight)):
        return None
    widths = owner._qwen38_horizontal_widths
    flat = torch.ops.vllm.qwen38_horizontal_gemm(x, weight, list(widths))
    return tuple(t.view(x.shape[0], n) for t, n in zip(
        flat.split([x.shape[0] * n for n in widths]), widths))


def prepare_horizontal_gemms(module: nn.Module) -> None:
    """Called by the loader's model-level post-load hook, before graph capture.

    Keep checkpoint Parameter objects/metadata for fallback and reloading, but
    rebind their storage to views of the cat to avoid duplicating dense weights.
    The buffer is rebuilt after each load and is not checkpointed.
    Only known direct consumers are enrolled. In particular no MoE callback
    caches input-dependent results across an opaque/custom-op boundary.
    """
    if _os.environ.get("QWEN38_W8_DENSE", "").strip().lower() not in ("", "0", "off"):
        from .w8_dense import report_storage
        report_storage(module)
    if _TRITON_SKINNY and _GLUON_THIN and not envs.VLLM_BATCH_INVARIANT:
        from .gluon_thin import MODE, prepare
        if MODE in ("1", "exact"):
            weights = []
            for child in module.modules():
                if type(getattr(child, "quant_method", None)) not in (
                        Qwen38NextLowLatencyLinearMethod, Qwen38NextLowLatencyEmbeddingMethod):
                    continue
                weight = child.weight
                weights.append(weight)
                if (isinstance(child, ParallelLMHead) and tuple(weight.shape) == (248320, 2560)
                        and _os.getenv("VLLM_MTP_DRAFT_VOCAB_RANGES")):
                    from .draft_vocab import parse_ranges
                    weights.extend(weight[a:b] for a, b in parse_ranges(
                        _os.environ["VLLM_MTP_DRAFT_VOCAB_RANGES"], weight.shape[0]))
            prepare(weights)
    if not (_TRITON_SKINNY and _HORIZONTAL) or envs.VLLM_BATCH_INVARIANT:
        return
    for owner in list(module.modules()):
        pairs = (("in_proj_qkvz", "in_proj_ba"), ("key_proj", "value_proj"))
        for names in pairs:
            children = [getattr(owner, name, None) for name in names]
            if not all(isinstance(c, LinearBase) for c in children):
                continue
            # Exclude quantized, LoRA and unsupported-device linears. The
            # constructor's skinny enrollment is the authority on eligibility.
            if not all(type(c.quant_method) is Qwen38NextLowLatencyLinearMethod
                       and c.bias is None for c in children):
                continue
            weights = [c.weight for c in children]
            if not all(_is_packed_row_major(w) and w.dtype == torch.bfloat16
                       and w.is_cuda and w.device == weights[0].device
                       and w.shape[1] == weights[0].shape[1] for w in weights):
                continue
            widths = tuple(w.shape[0] for w in weights)
            packed = torch.cat(weights, dim=0)
            owner.register_buffer("_qwen38_horizontal_weight", packed, persistent=False)
            for child, view in zip(children, packed.split(widths, dim=0)):
                child.weight.data = view
            owner._qwen38_horizontal_widths = widths

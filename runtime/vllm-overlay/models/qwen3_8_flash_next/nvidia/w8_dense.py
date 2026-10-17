# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental W8A16 dense projections using the installed Marlin kernels.

No activation quantization, CPU fallback or retained BF16 weight. Packed tensors
are runtime artifacts, not a reloadable checkpoint format; restart to refit.
"""

import logging

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from .w8_dense_policy import projection_scope, settings

log = logging.getLogger("vllm.qwen38.w8_dense")


@torch.no_grad()
def quantize_weight(weight, mode, group_size=-1):
    """Return encoded [N,K] bytes and BF16 [groups,N] scales.

    Quantize using the *rounded stored scale*. Bound temporary FP32 storage by
    processing 256 output channels at once. An all-zero group uses scale 1.
    INT8 stores signed [-127,127] plus the uint8b128 implicit bias of 128.
    """
    if (weight.ndim != 2 or weight.dtype != torch.bfloat16
            or mode not in ("fp8", "int8") or group_size not in (-1, 128)):
        raise ValueError("expected a BF16 matrix and fp8/int8, group -1/128")
    n, k = weight.shape
    width = k if group_size == -1 else group_size
    if k % width or k % 4:
        raise ValueError("K must divide the quantization group and byte pack")
    qmax = 448.0 if mode == "fp8" else 127.0
    encoded = torch.empty((n, k), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((n, k // width), dtype=weight.dtype, device=weight.device)
    for start in range(0, n, 256):
        w = weight[start:start + 256].float().reshape(-1, k // width, width)
        amax = w.abs().amax(dim=-1)
        s = (amax / qmax).clamp_min(torch.finfo(weight.dtype).tiny)
        s = torch.where(amax == 0, 1.0, s).to(weight.dtype)
        normalized = w / s.float().unsqueeze(-1)
        if mode == "fp8":
            q = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            q = (normalized.round().clamp(-127, 127) + 128).to(torch.uint8)
        encoded[start:start + 256].copy_(q.reshape(-1, k))
        scales[start:start + 256].copy_(s)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("nonfinite W8 scales/weights")
    # Reject NaN/Inf in the source, including NaN values hidden by INT8 casts.
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("cannot quantize nonfinite weights")
    return encoded, scales.T.contiguous()


def dequantize_weight(encoded, scales, mode, group_size=-1):
    """Unpacked round-trip reference for tests, never used in serving."""
    n, k = encoded.shape
    width = k if group_size == -1 else group_size
    values = (encoded.view(torch.float8_e4m3fn).float() if mode == "fp8"
              else encoded.float() - 128)
    return (values.reshape(n, -1, width) * scales.T.float().unsqueeze(-1)).reshape(n, k)


class PackedW8(nn.Module):
    def __init__(self, weight, mode, group_size=-1):
        super().__init__()
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (8, 0):
            raise ValueError("QWEN38 W8 dense currently requires CUDA SM80")
        if getattr(weight, "_vllm_is_uva_offloaded", False):
            raise ValueError("W8 dense requires resident weights (disable dense offload)")
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new, marlin_permute_scales,
            verify_marlin_supported, verify_marlin_supports_shape,
        )
        from vllm.scalar_type import scalar_types

        self.n, self.k = weight.shape
        self.mode, self.group_size = mode, group_size
        wtype = (scalar_types.float8_e4m3fn if mode == "fp8" else scalar_types.uint8b128)
        verify_marlin_supported(wtype, group_size, has_zp=False)
        verify_marlin_supports_shape(self.n, self.k, self.k, group_size)
        encoded, scales = quantize_weight(weight, mode, group_size)
        # Same GPTQ byte packing as pack_fp8_to_int32(size_k_first=False).
        # Each int32 holds four consecutive K bytes, then transpose to [K/4,N].
        if mode == "fp8":
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                pack_fp8_to_int32,
            )
            packed = pack_fp8_to_int32(encoded.view(torch.float8_e4m3fn), False).T.contiguous()
        else:
            packed = encoded.view(torch.int32).T.contiguous()
        empty = torch.empty(0, dtype=torch.int32, device=weight.device)
        qweight = ops.gptq_marlin_repack(
            b_q_weight=packed, perm=empty, size_k=self.k, size_n=self.n, num_bits=8)
        scales = marlin_permute_scales(scales, self.k, self.n, group_size)
        if mode == "fp8":
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                fp8_fused_exponent_bias_into_scales,
            )
            scales = fp8_fused_exponent_bias_into_scales(scales)
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("Marlin FP8 exponent-adjusted scales overflowed")
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("empty", empty)
        self.register_buffer("workspace", marlin_make_workspace_new(weight.device), persistent=False)

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers())

    def forward(self, x, bias=None):
        # Shapes/dtypes are metadata: no data-dependent host work during capture.
        if (x.dtype != torch.bfloat16 or x.shape[-1] != self.k
                or x.device != self.qweight.device):
            raise ValueError("W8 dense expects BF16 activations on the weight device")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.n))
        # Keep bias in logical order; adding after GEMM preserves skip_bias_add
        # and RowParallelLinear's rank-zero-only bias contract without repacking it.
        if self.mode == "fp8":
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                apply_fp8_marlin_linear,
            )
            result = apply_fp8_marlin_linear(
                x, self.qweight, self.scales, self.workspace, self.n, self.k,
                bias=None, input_dtype=None, use_fp32_reduce=True)
        else:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                apply_gptq_marlin_linear,
            )
            from vllm.scalar_type import scalar_types
            result = apply_gptq_marlin_linear(
                x, self.qweight, self.scales, self.empty, self.empty, self.empty,
                self.workspace, scalar_types.uint8b128, self.n, self.k,
                is_k_full=True, bias=None, input_dtype=None, use_fp32_reduce=True)
        return result if bias is None else result + bias


class Qwen38W8LinearMethod(UnquantizedLinearMethod):
    """Original checkpoint loaders write BF16; finalize replaces that storage."""

    def __init__(self, config, name):
        self.config, self.name = config, name

    def process_weights_after_loading(self, layer):
        if hasattr(layer, "_qwen38_w8"):
            return
        weight = layer.weight
        before = weight.numel() * weight.element_size()
        packed = PackedW8(weight, self.config.mode, self.config.group_size)
        layer.add_module("_qwen38_w8", packed)
        del layer.weight
        # Do not retain Parameter/data references on the method or in closures.
        del weight
        log.warning("W8_DENSE %s mode=%s group=%d bf16_bytes=%d w8_bytes=%d freed_bytes=%d",
                    self.name, self.config.mode, self.config.group_size, before,
                    packed.storage_bytes, before - packed.storage_bytes)

    def apply(self, layer, x, bias=None):
        return layer._qwen38_w8(x, bias)


def enroll(module, dtype):
    config = settings()
    if config is None:
        return
    # This experiment changes target weights only; MTP retains its prior policy.
    if type(module).__name__ == "Qwen3_8FlashNextMTP":
        return
    if dtype != torch.bfloat16:
        raise ValueError("QWEN38_W8_DENSE requires a BF16 model")
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("QWEN38_W8_DENSE requires SM80")
    selected = []
    for name, child in module.named_modules():
        if not isinstance(child, LinearBase):
            continue
        weight = getattr(child, "weight", None)
        if weight is None:
            continue
        scope = projection_scope(name, weight.shape)
        if scope not in config.scopes:
            continue
        # Prevent a double enrollment, LoRA wrapping or already quantized input.
        if type(child.quant_method) is not UnquantizedLinearMethod:
            raise ValueError(f"W8 selected {name} has a non-BF16 quant method")
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"W8 selected {name} is not BF16")
        child.quant_method = Qwen38W8LinearMethod(config, name)
        selected.append((name, scope, tuple(weight.shape)))
    if not selected:
        raise ValueError("QWEN38_W8_DENSE selected no target PP24 projections")
    log.warning("W8_DENSE changes target numerics: enrolled %d projections, mode=%s "
                "group=%d scopes=%s; draft/control/HC/router remain BF16",
                len(selected), config.mode, config.group_size, sorted(config.scopes))


def report_storage(module):
    """Post-finalize model/rank total; tensor storage, not allocator reserve."""
    packed = [child for child in module.modules() if isinstance(child, PackedW8)]
    if not packed:
        return
    before = sum(2 * child.n * child.k for child in packed)
    after = sum(child.storage_bytes for child in packed)
    log.warning("W8_DENSE_TOTAL layers=%d bf16_bytes=%d w8_bytes=%d freed_bytes=%d "
                "freed_GiB=%.6f (live tensor storage; allocator reserve may persist)",
                len(packed), before, after, before - after, (before - after) / 1024**3)

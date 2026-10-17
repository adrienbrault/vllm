# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next decode-step small-kernel fusions (all opt-in, all OFF by default).

Every knob below defaults to off so an untouched environment reproduces the
current behaviour bit-for-bit:

- ``QWEN38_FUSE_SIGMOID_MUL=1``: one Triton kernel for the decode-only
  ``x * torch.sigmoid(gate)`` epilogues (QSA attention output gate in
  ``models/qwen3_8_flash_next/nvidia/qsa.py`` and the shared-expert gate in
  ``model_executor/models/qwen2_moe.py``), replacing a sigmoid kernel plus a
  mul kernel with one launch.  The kernel reproduces the two-step rounding
  chain: sigmoid is computed in FP32 and rounded to BF16, the product is
  computed in FP32 and rounded to BF16, exactly like the two
  ``vectorized_elementwise`` kernels it replaces.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

QWEN38_FUSE_SIGMOID_MUL = os.environ.get("QWEN38_FUSE_SIGMOID_MUL", "0") == "1"


# ---------------------------------------------------------------------------
# 1. Fused sigmoid-mul epilogue
# ---------------------------------------------------------------------------
@triton.jit
def _qwen38_sigmoid_mul_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    DIM: tl.constexpr,
    GATE_BCAST: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    if GATE_BCAST:
        # Shared-expert gate shape [N, 1]: one scalar per token row.
        g = tl.load(g_ptr + row)
        g = tl.full([BLOCK], 0.0, tl.float32) + g
    else:
        g = tl.load(g_ptr + row * DIM + offs, mask, other=0.0)

    x = tl.load(x_ptr + row * DIM + offs, mask, other=0.0)

    # Step 1 mirrors torch.sigmoid(gate): FP32 compute, BF16 store.
    g_bf = tl.sigmoid(g.to(tl.float32)).to(y_ptr.dtype.element_ty)
    # Step 2 mirrors flat_output * sigmoid_out: BF16->FP32 mul, BF16 store.
    y = (x.to(tl.float32) * g_bf.to(tl.float32)).to(y_ptr.dtype.element_ty)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * DIM + offs, y, mask)


def _sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x * torch.sigmoid(gate)`` with the two-step BF16 rounding chain of
    the unfused pair of elementwise kernels."""
    assert x.dim() == 2 and x.stride(1) == 1 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float16)
    assert gate.dtype == x.dtype
    n, dim = x.shape
    assert gate.shape == (n, dim) or gate.shape == (n, 1)
    assert gate.is_contiguous()

    y = x.new_empty(x.shape)
    _qwen38_sigmoid_mul_kernel[(n,)](
        x,
        gate,
        y,
        DIM=dim,
        GATE_BCAST=gate.shape[1] == 1,
        BLOCK=triton.next_power_of_2(dim),
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return y


def _sigmoid_mul_fake(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    del gate
    return x.new_empty(x.shape)


direct_register_custom_op(
    op_name="qwen3_8_flash_next_sigmoid_mul",
    op_func=_sigmoid_mul,
    fake_impl=_sigmoid_mul_fake,
)


def qwen38_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.qwen3_8_flash_next_sigmoid_mul(x, gate)


def maybe_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Flag-gated dispatch used by the QSA output gate and shared expert."""
    if (
        QWEN38_FUSE_SIGMOID_MUL
        and x.dim() == 2
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float16)
        and gate.dtype == x.dtype
        and gate.is_contiguous()
        and gate.dim() == 2
        and gate.shape[0] == x.shape[0]
        and gate.shape[1] in (1, x.shape[1])
        and x.shape[1] <= 16384
    ):
        return qwen38_sigmoid_mul(x, gate)
    return x * torch.sigmoid(gate)

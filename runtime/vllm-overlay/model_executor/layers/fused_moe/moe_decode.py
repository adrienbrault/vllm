# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch routing for the opt-in sm_80 Humming indexed path.

Keep Humming's GEMM tuning, K accumulation and all BF16 stores unchanged.
Only integer metadata construction and the independent output tile geometry
change. No host reads, atomics, cross-CTA synchronization or persistent buffers.
"""

import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum_kernel,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _decode_align_kernel(
    TopIds,
    ExpertMap,
    SortedIds,
    ExpertIds,
    NumPadded,
    PAIRS: tl.constexpr,
    EXPERTS: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SORT_SIZE: tl.constexpr,
    FILL_SIZE: tl.constexpr,
):
    # The whole route list fits in one CTA (at most 160 pairs / 256 keys).
    i = tl.arange(0, SORT_SIZE)
    expert = tl.load(TopIds + i, i < PAIRS, other=EXPERTS)
    valid = (i < PAIRS) & (expert >= 0) & (expert < EXPERTS)
    expert = expert.to(tl.int32)
    if HAS_MAP:
        local = tl.load(ExpertMap + expert, valid, other=-1)
        valid = valid & (local >= 0)
    # Sort by GLOBAL expert, then flattened pair id. Retain separate blocks
    # when an expert_map aliases two global ids, just like the original path.
    key = tl.where(valid, expert * SORT_SIZE + i, EXPERTS * SORT_SIZE)
    key = tl.sort(key, descending=False)
    expert = key // SORT_SIZE
    pair = key % SORT_SIZE
    valid = expert < EXPERTS
    previous = tl.gather(expert, tl.maximum(i - 1, 0), axis=0)
    first = (i == 0) | (expert != previous)
    start = tl.associative_scan(tl.where(first, i, 0), axis=0,
                                combine_fn=_maximum)
    rank = i - start
    new_block = valid & (rank % BLOCK_M == 0)
    block = tl.cumsum(new_block.to(tl.int32), axis=0) - 1
    num_blocks = tl.sum(new_block.to(tl.int32), axis=0)

    # Fully initialize padding and the unused tail: Humming may prefetch
    # metadata beyond NumPadded. One CTA owns every store. The barrier is
    # essential: no lane may scatter a real id before padding stores finish.
    fill = tl.arange(0, FILL_SIZE)
    tl.store(SortedIds + fill, PAIRS, fill < PAIRS * BLOCK_M)
    tl.store(ExpertIds + i, -1, i < PAIRS)
    tl.debug_barrier()
    tl.store(SortedIds + block * BLOCK_M + rank % BLOCK_M, pair, valid)
    if HAS_MAP:
        expert = tl.load(ExpertMap + expert, valid, other=-1).to(tl.int32)
    tl.store(ExpertIds + block, expert, new_block)
    tl.store(NumPadded, num_blocks * BLOCK_M)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


def align_decode(topk_ids: torch.Tensor, expert_map: torch.Tensor | None,
                 block_size: int = 16):
    """Equivalent indexed-GEMM metadata, with stable ascending pair order.

    Caller restricts dispatch to E=512, top-k=10, 1<=M<=16, block_size=16.
    The allocation sizes match moe_align_block_size for this geometry.
    """
    assert block_size == 16
    assert topk_ids.ndim == 2 and topk_ids.shape[1] == 10
    assert 1 <= topk_ids.shape[0] <= 16 and topk_ids.is_contiguous()
    pairs = topk_ids.numel()
    sorted_ids = torch.empty((pairs * block_size,), device=topk_ids.device,
                             dtype=torch.int32)
    expert_ids = torch.empty((pairs,), device=topk_ids.device, dtype=torch.int32)
    num_padded = torch.empty((1,), device=topk_ids.device, dtype=torch.int32)
    _decode_align_kernel[(1,)](
        topk_ids, expert_map, sorted_ids, expert_ids, num_padded,
        PAIRS=pairs, EXPERTS=512, HAS_MAP=expert_map is not None,
        BLOCK_M=block_size, SORT_SIZE=triton.next_power_of_2(pairs),
        FILL_SIZE=triton.next_power_of_2(pairs * block_size), num_warps=4,
    )
    return sorted_ids, expert_ids, num_padded


def combine_decode(inputs: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, expert_map: torch.Tensor | None,
                   outputs: torch.Tensor):
    """Retile the EXISTING reduction kernel: same ten ordered FP32 updates.

    40 CTAs at M=4, 160 at M=16, instead of 12/24. Each CTA owns a
    disjoint output interval. Retain the original compiler's FMA policy.
    """
    m, top_k, hidden = inputs.shape
    assert top_k == 10 and hidden == 2560 and 1 <= m <= 16
    moe_fused_mul_sum_kernel[(triton.cdiv(hidden, 256), m)](
        inputs, topk_weights, outputs, topk_ids, expert_map, m,
        top_k * hidden, expert_map is not None, top_k, hidden, 1, 256,
        num_warps=4, num_stages=4,
    )
    return outputs

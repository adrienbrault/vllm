# SPDX-License-Identifier: Apache-2.0
"""Experimental skinny-M BF16 GEMM for decode (M <= 8; opt-in M <= 16), Triton tensor cores.

Pads M to 16 rows and streams each weight byte once; FP32 accumulation, BF16
output; deterministic split-K (fixed-order reduction). Used only through the
existing Qwen3.8 low-latency GEMM hook when QWEN38_TRITON_SKINNY=1.
"""
import math
import os
import torch
import triton
import triton.language as tl

MAX_M = 16 if os.getenv("QWEN38_SKINNY_M16", "0") == "1" else 8
_FUSED_REDUCE = os.getenv("QWEN38_SKINNY_FUSED_REDUCE", "0") == "1"
_SM80_STAGES = os.getenv("QWEN38_SKINNY_SM80_STAGES", "0") == "1"
_GLUON_THIN = os.getenv("QWEN38_GLUON_THIN", "0") in ("1", "exact")
_workspace = {}
_configs = {}


@triton.jit
def _reduce_kernel(part_ptr, out_ptr, M, N, stride_om, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * (BLOCK_M * N) + offs_m[:, None] * N + offs_n[None, :],
                       mask=mask, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _skinny_tile(x_ptr, w_ptr, out_ptr, part_ptr, counter_ptr, M, N, K,
                 pid_n, pid_k, FUSED_REDUCE: tl.constexpr,
                       stride_xm, stride_wn, stride_om,
                       K_PER_SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, STAGES: tl.constexpr):
    # Tensor-core variant: M padded to 16 rows; W tile [BLOCK_N, BLOCK_K].
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER_SPLIT
    for k0 in tl.range(0, K_PER_SPLIT, BLOCK_K, num_stages=STAGES):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                    eviction_policy='evict_first')
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    if SPLIT_K == 1:
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
    else:
        base = part_ptr + pid_k * (16 * N)
        tl.store(base + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=m_mask[:, None] & n_mask[None, :])
        if FUSED_REDUCE:
            # Every thread publishes its partial before a single CTA arrival.
            # No CTA waits: the last arrival alone reduces in the old split order.
            fence = tl.inline_asm_elementwise(
                "membar.gl; mov.u32 $0, 0;", constraints="=r,r",
                args=[tl.arange(0, 128)], dtype=tl.int32,
                is_pure=False, pack=1)
            tl.debug_barrier()
            arrived = tl.atomic_add(counter_ptr + pid_n, 1 + tl.sum(fence, 0),
                                    sem="acq_rel", scope="gpu")
            if arrived == SPLIT_K - 1:
                total = tl.zeros((16, BLOCK_N), tl.float32)
                for s in range(SPLIT_K):
                    total += tl.load(
                        part_ptr + s * (16 * N) + offs_m[:, None] * N + offs_n[None, :],
                        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
                        cache_modifier=".cg")
                tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                         total.to(out_ptr.dtype.element_ty),
                         mask=m_mask[:, None] & n_mask[None, :])
                tl.debug_barrier()
                tl.atomic_xchg(counter_ptr + pid_n, 0, sem="release", scope="gpu")


@triton.jit
def _skinny_dot_kernel(x_ptr, w_ptr, out_ptr, part_ptr, counter_ptr, M, N, K,
                       stride_xm, stride_wn, stride_om,
                       K_PER_SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, STAGES: tl.constexpr,
                       FUSED_REDUCE: tl.constexpr = False):
    _skinny_tile(x_ptr, w_ptr, out_ptr, part_ptr, counter_ptr, M, N, K,
                 tl.program_id(0), tl.program_id(1), FUSED_REDUCE,
                 stride_xm, stride_wn, stride_om,
                 K_PER_SPLIT, BLOCK_N, BLOCK_K, SPLIT_K, STAGES)


@triton.jit
def _horizontal_section(x, w, out, part, counters, M, K: tl.constexpr, SX, SW,
                        N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        SPLIT: tl.constexpr, STAGES: tl.constexpr,
                        TILES: tl.constexpr, K_PER_SPLIT: tl.constexpr,
                        START: tl.constexpr, WEIGHT_OFFSET: tl.constexpr,
                        PART_OFFSET: tl.constexpr, COUNTER_OFFSET: tl.constexpr,
                        FUSED_REDUCE: tl.constexpr):
    pid = tl.program_id(0)
    if (pid >= START) & (pid < START + TILES * SPLIT):
        local = pid - START
        _skinny_tile(x, w + WEIGHT_OFFSET * SW, out + WEIGHT_OFFSET * M,
                     part + PART_OFFSET, counters + COUNTER_OFFSET, M, N, K,
                     local % TILES, local // TILES, FUSED_REDUCE,
                     SX, SW, N, K_PER_SPLIT, BN, BK, SPLIT, STAGES)


@triton.jit
def _horizontal_kernel(x, w, out, part, counters, M, K: tl.constexpr, SX, SW,
                       N0: tl.constexpr, BN0: tl.constexpr, BK0: tl.constexpr,
                       SPLIT0: tl.constexpr, STAGES0: tl.constexpr, TILES0: tl.constexpr,
                       K_PER_SPLIT0: tl.constexpr, START0: tl.constexpr, WEIGHT_OFFSET0: tl.constexpr,
                       PART_OFFSET0: tl.constexpr, COUNTER_OFFSET0: tl.constexpr, REDUCE_START0: tl.constexpr,
                       N1: tl.constexpr, BN1: tl.constexpr, BK1: tl.constexpr,
                       SPLIT1: tl.constexpr, STAGES1: tl.constexpr, TILES1: tl.constexpr,
                       K_PER_SPLIT1: tl.constexpr, START1: tl.constexpr, WEIGHT_OFFSET1: tl.constexpr,
                       PART_OFFSET1: tl.constexpr, COUNTER_OFFSET1: tl.constexpr, REDUCE_START1: tl.constexpr,
                       N2: tl.constexpr, BN2: tl.constexpr, BK2: tl.constexpr,
                       SPLIT2: tl.constexpr, STAGES2: tl.constexpr, TILES2: tl.constexpr,
                       K_PER_SPLIT2: tl.constexpr, START2: tl.constexpr, WEIGHT_OFFSET2: tl.constexpr,
                       PART_OFFSET2: tl.constexpr, COUNTER_OFFSET2: tl.constexpr, REDUCE_START2: tl.constexpr,
                       FUSED_REDUCE: tl.constexpr):
    # Fixed scalar arity avoids tuple flattening and unrolled constexpr
    # redeclarations in Triton 3.7.1. Unused sections are compiled away.
    if N0 > 0:
        _horizontal_section(x, w, out, part, counters, M, K, SX, SW,
                            N0, BN0, BK0, SPLIT0, STAGES0, TILES0, K_PER_SPLIT0,
                            START0, WEIGHT_OFFSET0, PART_OFFSET0, COUNTER_OFFSET0,
                            FUSED_REDUCE)
    if N1 > 0:
        _horizontal_section(x, w, out, part, counters, M, K, SX, SW,
                            N1, BN1, BK1, SPLIT1, STAGES1, TILES1, K_PER_SPLIT1,
                            START1, WEIGHT_OFFSET1, PART_OFFSET1, COUNTER_OFFSET1,
                            FUSED_REDUCE)
    if N2 > 0:
        _horizontal_section(x, w, out, part, counters, M, K, SX, SW,
                            N2, BN2, BK2, SPLIT2, STAGES2, TILES2, K_PER_SPLIT2,
                            START2, WEIGHT_OFFSET2, PART_OFFSET2, COUNTER_OFFSET2,
                            FUSED_REDUCE)


@triton.jit
def _horizontal_reduce_section(part, out, M, N: tl.constexpr,
                               SPLIT: tl.constexpr, OUT_OFFSET: tl.constexpr,
                               PART_OFFSET: tl.constexpr, START: tl.constexpr):
    pid = tl.program_id(0)
    tiles: tl.constexpr = (N + 127) // 128
    if (pid >= START) & (pid < START + tiles):
        ns = (pid - START) * 128 + tl.arange(0, 128)
        ms = tl.arange(0, 16)
        mask = (ms[:, None] < M) & (ns[None, :] < N)
        acc = tl.zeros((16, 128), tl.float32)
        for s in range(SPLIT):
            acc += tl.load(part + PART_OFFSET + s * 16 * N + ms[:, None] * N + ns[None, :],
                           mask=mask, other=0.0)
        tl.store(out + OUT_OFFSET * M + ms[:, None] * N + ns[None, :],
                 acc.to(out.dtype.element_ty), mask=mask)


@triton.jit
def _horizontal_reduce(part, out, M,
                       N0: tl.constexpr, BN0: tl.constexpr, BK0: tl.constexpr,
                       SPLIT0: tl.constexpr, STAGES0: tl.constexpr, TILES0: tl.constexpr,
                       K_PER_SPLIT0: tl.constexpr, START0: tl.constexpr, WEIGHT_OFFSET0: tl.constexpr,
                       PART_OFFSET0: tl.constexpr, COUNTER_OFFSET0: tl.constexpr, REDUCE_START0: tl.constexpr,
                       N1: tl.constexpr, BN1: tl.constexpr, BK1: tl.constexpr,
                       SPLIT1: tl.constexpr, STAGES1: tl.constexpr, TILES1: tl.constexpr,
                       K_PER_SPLIT1: tl.constexpr, START1: tl.constexpr, WEIGHT_OFFSET1: tl.constexpr,
                       PART_OFFSET1: tl.constexpr, COUNTER_OFFSET1: tl.constexpr, REDUCE_START1: tl.constexpr,
                       N2: tl.constexpr, BN2: tl.constexpr, BK2: tl.constexpr,
                       SPLIT2: tl.constexpr, STAGES2: tl.constexpr, TILES2: tl.constexpr,
                       K_PER_SPLIT2: tl.constexpr, START2: tl.constexpr, WEIGHT_OFFSET2: tl.constexpr,
                       PART_OFFSET2: tl.constexpr, COUNTER_OFFSET2: tl.constexpr, REDUCE_START2: tl.constexpr):
    if SPLIT0 > 1:
        _horizontal_reduce_section(part, out, M, N0, SPLIT0,
                                   WEIGHT_OFFSET0, PART_OFFSET0, REDUCE_START0)
    if SPLIT1 > 1:
        _horizontal_reduce_section(part, out, M, N1, SPLIT1,
                                   WEIGHT_OFFSET1, PART_OFFSET1, REDUCE_START1)
    if SPLIT2 > 1:
        _horizontal_reduce_section(part, out, M, N2, SPLIT2,
                                   WEIGHT_OFFSET2, PART_OFFSET2, REDUCE_START2)


def _horizontal_args(sections):
    """Only plain scalar ints cross a JIT boundary; tuples stay on the host."""
    if not 1 <= len(sections) <= 3:
        raise ValueError("horizontal GEMM supports one to three sections")
    fields = ("N", "BN", "BK", "SPLIT", "STAGES", "TILES", "K_PER_SPLIT",
              "START", "WEIGHT_OFFSET", "PART_OFFSET", "COUNTER_OFFSET", "REDUCE_START")
    padded = (*sections, *((0,) * len(fields) for _ in range(3 - len(sections))))
    return {f"{name}{i}": int(value)
            for i, section in enumerate(padded) for name, value in zip(fields, section)}


def _horizontal_plan(widths, configs, k):
    """Immutable section descriptors; all offsets are computed before launch."""
    sections = []
    start = weight_offset = part_offset = counter_offset = reduce_start = 0
    for n, (bn, bk, split, stages) in zip(widths, configs):
        tiles = triton.cdiv(n, bn)
        k_per_split = triton.cdiv(triton.cdiv(k, split), bk) * bk
        sections.append((n, bn, bk, split, stages, tiles, k_per_split,
                         start, weight_offset, part_offset, counter_offset, reduce_start))
        start += tiles * split
        weight_offset += n
        if split > 1:
            part_offset += split * 16 * n
            counter_offset += tiles
            reduce_start += triton.cdiv(n, 128)
    return tuple(sections), start, reduce_start, part_offset, counter_offset


def config_for(n, k):
    key = (n, k)
    cfg = _configs.get(key)
    if cfg is not None:
        return cfg
    if n >= 2048:
        bk = 64 if k < 1024 else 128
        cfg = (32, bk, 1, 3 if n < 4096 else 2)
    else:
        bn, bk = 16, 128
        tiles = triton.cdiv(n, bn)
        split = 1
        target = 96.0 / tiles
        if target > 1.5:
            split = min(8, 1 << int(round(math.log2(target))))
        while split > 1 and k < 2 * bk * split:
            split //= 2
        cfg = (bn, bk, split, 4)
    _configs[key] = cfg
    return cfg


def execution_config(n, k, device):
    bn, bk, split, stages = config_for(n, k)
    if _SM80_STAGES and torch.cuda.get_device_capability(device) == (8, 0):
        # Scheduling only: preserve BN, BK and split, hence the MMA/add order.
        stages = min(stages, 2 if k <= 2560 else 3)
    return bn, bk, split, stages


def _scratch(x, size, counters, fused_reduce):
    # A capture gets private storage for EACH invocation. A global shape cache
    # would alias independent graphs replayed concurrently on different streams.
    # zeros adds a captured initialization node (memset or fill, torch-dependent);
    # the last CTA also resets counters. Eager calls reuse within one stream.
    capturing = torch.cuda.is_current_stream_capturing()
    key = (x.device, torch.cuda.current_stream(x.device).cuda_stream, size, counters)
    scratch = None if capturing else _workspace.get(key)
    if scratch is None:
        part = torch.empty(size, dtype=torch.float32, device=x.device)
        count = (torch.zeros(counters, dtype=torch.int32, device=x.device)
                 if fused_reduce else part)
        scratch = (part, count)
        if not capturing:
            _workspace[key] = scratch
    elif fused_reduce and scratch[1].dtype != torch.int32:
        scratch = (scratch[0], torch.zeros(counters, dtype=torch.int32, device=x.device))
        _workspace[key] = scratch
    return scratch


def skinny_mm(x, w, *, fused_reduce=None, gluon=True):
    if _GLUON_THIN and gluon and fused_reduce is None:
        from .gluon_thin import try_mm
        result = try_mm(x, w)
        if result is not None:
            return result
    m, k = x.shape
    n = w.shape[0]
    if not (1 <= m <= 16 and x.stride(1) == w.stride(1) == 1):
        raise ValueError("skinny_mm requires 1..16 rows and contiguous K")
    bn, bk, split, stages = execution_config(n, k, x.device)
    fused = _FUSED_REDUCE if fused_reduce is None else fused_reduce
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    k_per_split = triton.cdiv(triton.cdiv(k, split), bk) * bk
    part = count = out
    if split > 1:
        if fused:
            part, count = _scratch(x, split * 16 * n, triton.cdiv(n, bn), True)
        else:
            # Preserve the baseline allocation/cache policy when flags are off.
            key = (x.device, split * 16 * n)
            part = _workspace.get(key)
            if part is None:
                part = torch.empty(split * 16 * n, dtype=torch.float32, device=x.device)
                _workspace[key] = part
    _skinny_dot_kernel[(triton.cdiv(n, bn), split)](
        x, w, out, part, count, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        K_PER_SPLIT=k_per_split, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split,
        STAGES=stages, FUSED_REDUCE=fused, num_warps=4)
    if split > 1 and not fused:
        _reduce_kernel[(triton.cdiv(n, 128),)](
            part, out, m, n, out.stride(0), BLOCK_M=16,
            BLOCK_N=128, SPLIT_K=split, num_warps=4)
    return out


def horizontal_mm(x, weight, widths, *, fused_reduce=None):
    """Load-time cat of W along N; output sections are contiguous tensor views.

    Each section retains its own original config, including split boundaries.
    Never call config_for(sum(widths), K): that changes FP32 accumulation order.
    """
    m, k = x.shape
    widths = tuple(widths)
    if not (1 <= m <= 16 and 1 <= len(widths) <= 3 and min(widths) > 0
            and sum(widths) == weight.shape[0] and k == weight.shape[1]
            and x.stride(1) == weight.stride(1) == 1):
        raise ValueError("invalid horizontal GEMM shape/strides")
    configs = tuple(execution_config(n, k, x.device) for n in widths)
    sections, grid, reduce_tiles, size, tiles = _horizontal_plan(widths, configs, k)
    section_args = _horizontal_args(sections)
    fused = _FUSED_REDUCE if fused_reduce is None else fused_reduce
    out = torch.empty(m * sum(widths), dtype=x.dtype, device=x.device)
    part = count = out
    if size:
        part, count = _scratch(x, size, tiles, fused)
    _horizontal_kernel[(grid,)](
        x, weight, out, part, count, m, k, x.stride(0), weight.stride(0),
        **section_args, FUSED_REDUCE=fused, num_warps=4)
    if reduce_tiles and not fused:
        _horizontal_reduce[(reduce_tiles,)](
            part, out, m, **section_args, num_warps=4)
    return out

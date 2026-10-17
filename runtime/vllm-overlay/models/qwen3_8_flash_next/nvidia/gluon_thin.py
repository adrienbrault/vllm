# SPDX-License-Identifier: Apache-2.0
"""Measured TP1 thin GEMM dispatch. No import, JIT or scratch allocation in capture.

Output allocation follows the existing functional custom-op contract. Supplying
``out`` exercises an entirely allocation-free warmed launch. Scratch belongs to
the CUDA stream, so independently captured streams do not share arrival counters.
As with the model's input buffers, replay a graph on its owning stream.
"""
import importlib
import logging
import os

import torch

from .gluon_thin_table import TABLES

MODE = os.getenv("QWEN38_GLUON_THIN", "0")
_COMPATIBLE = (os.getenv("QWEN38_TRITON_SKINNY", "0") == "1"
               and os.getenv("VLLM_BATCH_INVARIANT", "0").lower() not in ("1", "true")
               and not any(os.getenv(name, "0") == "1" for name in (
                   "QWEN38_SKINNY_M16", "QWEN38_SKINNY_FUSED_REDUCE",
                   "QWEN38_SKINNY_SM80_STAGES", "QWEN38_SKINNY_HORIZONTAL")))
_backend = None
_devices = {}
_ready = set()
_failed = set()
_log = logging.getLogger(__name__)


def gate(mode, compatible, capability, sms, importable):
    """Fail closed on anything outside the measured device/software path."""
    return (mode in TABLES and compatible and capability == (8, 0)
            and sms == 74 and importable)


def _init_device(device):
    global _backend
    if device.index in _devices:
        return _devices[device.index]
    # This is called ONLY outside capture. Missing Gluon keeps the old path.
    try:
        props = torch.cuda.get_device_properties(device)
        ok = gate(MODE, _COMPATIBLE, (props.major, props.minor),
                  props.multi_processor_count, True)
        if ok and _backend is None:
            _backend = importlib.import_module(".morrow.thin_gemm_gluon", __package__)
            _backend._SERVED_M = (1, 2, 4, 8, 16, 28, 32)
            _backend._FORCE_M[0] = 0
            _backend._CONFIG_OVERRIDES.clear()
            for (n, k, m), cfg in TABLES[MODE].items():
                family, bm, bn, bk, split, warps, stages, wm, acq = cfg
                if family != "gemv":
                    row = (bn, bk, split, warps, stages)
                    if family != "triton":
                        row += ({"mma": wm, "sliced": 0, "balanced": -1}[family], acq)
                    _backend._CONFIG_OVERRIDES[n, k, m] = row
    except Exception as exc:
        ok = False
        _log.warning("QWEN38_GLUON_THIN unavailable; keeping current GEMM: %s", exc)
    _devices[device.index] = ok
    return ok


def _packed(t):
    return (t.ndim == 2 and t.is_cuda and t.dtype == torch.bfloat16
            and t.stride() == (t.shape[1], 1) and t.data_ptr() % 16 == 0)


def try_mm(x, weight, *, out=None):
    """Return None for fallback. A cold capture never imports, compiles or warms.

    Eager calls (including vLLM's pre-capture dummy runs) may warm a new stream.
    An entry is admitted only after its actual specialized launch succeeds.
    Exact mode is the measured torch.equal-to-current subset at this exact M.
    """
    if MODE not in TABLES or not _COMPATIBLE:
        return None
    if not (_packed(x) and _packed(weight) and x.device == weight.device
            and x.shape[1] == weight.shape[1]):
        return None
    m, k = x.shape
    n = weight.shape[0]
    if (n, k, m) not in TABLES[MODE]:
        return None
    if out is not None and not (_packed(out) and out.device == x.device
                                and out.shape == (m, n)):
        return None
    # CUDA current-stream APIs refer to the current device during capture.
    if x.device.index != torch.cuda.current_device():
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if x.device.index not in _devices:
        if capturing or not _init_device(x.device):
            return None
    if not _devices[x.device.index]:
        return None
    stream = torch.cuda.current_stream(x.device).cuda_stream
    key = (x.device.index, stream, n, k, m)
    if key in _failed or (capturing and key not in _ready):
        return None
    # Keep distinct outputs for independent functional calls in the graph.
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    with _backend.workspace_lane(stream):
        if key not in _ready:
            try:
                _backend.thin_gemm(x, weight, out=out)
                torch.cuda.current_stream(x.device).synchronize()
            except Exception as exc:
                _failed.add(key)
                _log.warning("QWEN38_GLUON_THIN warmup failed for %s; fallback: %s", key, exc)
                return None
            _ready.add(key)
        else:
            _backend.thin_gemm(x, weight, out=out)
    return out


def prepare(weights):
    """Post-load warmup using the real weights, including restricted head views."""
    if MODE not in TABLES or not _COMPATIBLE:
        return
    seen = set()
    for weight in weights:
        if not _packed(weight):
            continue
        n, k = weight.shape
        identity = (weight.device.index, n, k)
        if identity in seen:
            continue
        seen.add(identity)
        with torch.cuda.device(weight.device):
            if torch.cuda.is_current_stream_capturing() or not _init_device(weight.device):
                continue
            for tn, tk, m in TABLES[MODE]:
                if (tn, tk) == (n, k):
                    x = torch.zeros((m, k), dtype=torch.bfloat16, device=weight.device)
                    try_mm(x, weight)

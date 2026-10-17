# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP8 decode lookup; all tensor pointers are bound at registration.

Build explicitly at image build time. Loading this module or serving a request
never invokes a compiler. The C ABI uses only the Python standard library.
"""

import ctypes as C
import functools
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import weakref


logger = logging.getLogger(__name__)


class _Binding(C.Structure):
    _fields_ = [
        (name, C.c_int64)
        for name in (
            "ngram", "heads", "dim", "shard_rows", "shards_count", "targets_count",
            "max_tokens", "max_reqs", "eos", "clamp_inputs", "timeout_ns",
        )
    ] + [
        (name, C.c_void_p)
        for name in (
            "input", "starts", "context", "multipliers", "sizes", "offsets",
            "shards", "rows", "outputs", "ready", "consumed",
            "fds", "bases", "reader",
        )
    ]


class _Stats(C.Structure):
    _fields_ = [(name, C.c_int64) for name in (
        "rows", "nonresident", "unknown", "nowait_hits", "deferred",
        "wall_ns", "wait_ns", "read_ns",
    )]


def library_path() -> Path:
    return Path(os.environ.get("QWEN38_PLE_FAST_LIBRARY", "/opt/qwen38/ple_fast.so"))


def build_library(output: str | Path | None = None) -> Path:
    """Explicit build helper for Docker and CPU tests, never called by serving."""
    output = Path(output) if output is not None else library_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [*shlex.split(os.environ.get("CXX", "c++")), "-O3", "-std=c++17",
         "-fPIC", "-shared", "-pthread", str(Path(__file__).with_name("ple_fast.cpp")),
         "-o", str(output)],
        check=True,
    )
    return output


@functools.cache
def _library():
    path = library_path()
    if not path.is_file():
        raise RuntimeError(
            f"QWEN38_PLE_FAST=1 requires the image-built {path}; rebuild the image "
            "or explicitly build fast.build_library() before serving"
        )
    library = C.CDLL(str(path))  # releases the GIL during lookup / consumed wait
    version = getattr(library, "ple_fast_abi_version", None)
    if version is None or version() != 2:
        raise RuntimeError(f"Stale PLE fast library {path}; rebuild it for ABI 2")
    library.ple_fast_run.argtypes = [C.POINTER(_Binding), C.c_int64, C.c_int64,
                                    C.c_int, C.POINTER(_Stats)]
    library.ple_fast_run.restype = C.c_int
    library.ple_fast_create.argtypes = [C.c_int64, C.c_int]
    library.ple_fast_create.restype = C.c_void_p
    library.ple_fast_destroy.argtypes = [C.c_void_p]
    library.ple_fast_destroy.restype = None
    return library


def _close_reader(library, pointer, fds):
    library.ple_fast_destroy(pointer)
    for fd in fds:
        os.close(fd)


class _Reader:
    """Registration-owned scratch, descriptors and at most eight sleeping threads.

    One caller per reader; close only after its native call has returned.
    """

    def __init__(self, capacity, paths=(), threads=None):
        library = _library()
        workers = min(8, os.cpu_count() or 1) if threads is None else threads
        opened = {}
        try:
            for path in paths:
                if path not in opened:
                    opened[path] = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            pointer = library.ple_fast_create(capacity, workers)
            if not pointer:
                raise RuntimeError("Cannot create PLE fast reader")
        except BaseException:
            for fd in opened.values():
                os.close(fd)
            raise
        self.pointer = pointer
        self.fds = (C.c_int32 * len(paths))(*(opened[p] for p in paths)) if paths else None
        self._finalize = weakref.finalize(self, _close_reader, library, pointer,
                                         tuple(opened.values()))

    def close(self):
        self._finalize()


class BoundDecode:
    """Own the registered buffers and one allocation-free native call descriptor."""

    def __init__(self, layer, inputs, targets, clamp_inputs: bool, timeout: float):
        import torch

        def check(tensor, dtype, ndim):
            if (tensor is None or tensor.device.type != "cpu"
                    or tensor.dtype != dtype or tensor.ndim != ndim
                    or not tensor.is_contiguous()):
                raise ValueError("PLE fast binding requires contiguous CPU tensors")

        check(inputs.input_ids_buf, torch.int32, 1)
        check(inputs.query_start_loc_buf, torch.int32, 1)
        check(inputs.ngram_context_buf, torch.int32, 2)
        if inputs.ngram_context_buf.shape[1] != layer.ngram_size - 1:
            raise ValueError("PLE fast n-gram context width mismatch")
        for tensor in (layer.layer_multipliers, layer.ngram_heads_vocab_sizes,
                       layer.ngram_heads_offsets):
            check(tensor, torch.int64, 1)
        heads = (layer.ngram_size - 1) * layer.heads_per_ngram
        if (layer.layer_multipliers.numel() != layer.ngram_size
                or layer.ngram_heads_vocab_sizes.numel() != heads
                or layer.ngram_heads_offsets.numel() != heads
                or layer.embedding_dim != heads * layer.head_dim
                or bool((layer.ngram_heads_vocab_sizes <= 0).any())
                or bool((layer.ngram_heads_offsets < 0).any())):
            raise ValueError("PLE fast hash layout mismatch")
        table = layer._ple_mmap
        if table is not None:
            views, shard_rows = table.views, table.shard_rows
            for view in views:
                check(view, torch.uint8, 2)
        else:
            weight = layer.ngram_embedding.weight
            check(weight, torch.float8_e4m3fn, 2)
            views, shard_rows = [weight], weight.shape[0]
        if not views or shard_rows <= 0 or any(
            v.shape[1] != layer.head_dim or v.shape[0] > shard_rows for v in views
        ):
            raise ValueError("PLE fast table layout mismatch")
        for target in targets:
            check(target.cpu_output_buffer, torch.float8_e4m3fn, 2)
            check(target.ready_flag, torch.int32, 1)
            check(target.consumed_flag, torch.int32, 1)
            if (target.cpu_output_buffer.shape[1] != layer.embedding_dim
                    or target.ready_flag.numel() != 1
                    or target.consumed_flag.numel() != 1):
                raise ValueError("PLE fast output layout mismatch")
        if not targets:
            raise ValueError("PLE fast requires an output target")
        shards = (C.c_void_p * len(views))(*(v.data_ptr() for v in views))
        rows = (C.c_int64 * len(views))(*(v.shape[0] for v in views))
        outputs = (C.c_void_p * len(targets))(
            *(t.cpu_output_buffer.data_ptr() for t in targets))
        ready = (C.c_void_p * len(targets))(*(t.ready_flag.data_ptr() for t in targets))
        consumed = (C.c_void_p * len(targets))(
            *(t.consumed_flag.data_ptr() for t in targets))
        self._reader = _Reader(64 * heads, table.paths if table is not None else ())
        bases = (C.c_int64 * len(views))(*table.bases) if table is not None else None
        self._owners = (layer, inputs, targets, views, shards, rows, outputs, ready, consumed, bases)
        self._binding = _Binding(
            layer.ngram_size, layer.heads_per_ngram, layer.head_dim, shard_rows,
            len(views), len(targets),
            min(inputs.input_ids_buf.numel(), layer.positions_buffer.numel(),
                *(t.cpu_output_buffer.shape[0] for t in targets)),
            min(inputs.ngram_context_buf.shape[0], inputs.query_start_loc_buf.numel() - 1,
                layer.padded_buffer.shape[0]),
            layer.eos_token_id, clamp_inputs, int(timeout * 1_000_000_000),
            inputs.input_ids_buf.data_ptr(), inputs.query_start_loc_buf.data_ptr(),
            inputs.ngram_context_buf.data_ptr(), layer.layer_multipliers.data_ptr(),
            layer.ngram_heads_vocab_sizes.data_ptr(), layer.ngram_heads_offsets.data_ptr(),
            C.addressof(shards), C.addressof(rows), C.addressof(outputs),
            C.addressof(ready), C.addressof(consumed),
            C.addressof(self._reader.fds) if self._reader.fds is not None else None,
            C.addressof(bases) if bases is not None else None, self._reader.pointer,
        )
        self._pointer = C.pointer(self._binding)
        self._run = _library().ple_fast_run
        self.last_stats = _Stats()
        self._stats_pointer = C.pointer(self.last_stats)
        self._log_every = max(0, int(os.environ.get("QWEN38_PLE_FAST_LOG_EVERY", "100")))
        self._calls = 0
        self._summary = {}
        self._label = f"input=0x{inputs.input_ids_buf.data_ptr():x}"

    def close(self):
        self._reader.close()

    def _record(self, tokens, reqs):
        if not self._log_every:
            return
        key = (tokens, reqs)
        if key not in self._summary:
            self._summary[key] = [0] * 15
        totals = self._summary[key]
        stats = self.last_stats
        totals[0] += 1
        for i, (name, _) in enumerate(_Stats._fields_, 1):
            totals[i] += getattr(stats, name)
        totals[9] = max(totals[9], stats.wall_ns)
        totals[10] = max(totals[10], stats.wait_ns)
        if stats.nonresident:
            totals[11] += 1
            totals[12] += stats.wall_ns
        elif stats.unknown:
            totals[13] += 1
            totals[14] += stats.wall_ns
        self._calls += 1
        if self._calls % self._log_every:
            return
        for (nt, nr), values in self._summary.items():
            (n, rows, cold, unknown, hits, deferred, wall, wait, read,
             max_wall, max_wait, cold_n, cold_wall, unknown_n, unknown_wall) = values
            warm_n = n - cold_n - unknown_n
            logger.info("[PLE_FAST] %s %s", self._label, json.dumps(dict(
                calls=n, tokens=nt, reqs=nr, rows=rows, nonresident=cold,
                residency_unknown=unknown, nowait_hits=hits, deferred_rows=deferred,
                wall_us=wall / n / 1000, wall_max_us=max_wall / 1000,
                consumed_wait_us=wait / n / 1000, consumed_wait_max_us=max_wait / 1000,
                read_us=read / n / 1000, cold_calls=cold_n,
                warm_calls=warm_n, unknown_calls=unknown_n,
                unknown_wall_us=unknown_wall / unknown_n / 1000 if unknown_n else None,
                cold_wall_us=cold_wall / cold_n / 1000 if cold_n else None,
                warm_wall_us=(wall - cold_wall - unknown_wall) / warm_n / 1000 if warm_n else None,
            ), sort_keys=True))
        self._summary.clear()

    def residency(self, num_tokens: int, num_reqs: int) -> _Stats:
        """Non-faulting snapshot for cold-cache tests; no output/flag writes."""
        if not self._reader._finalize.alive:
            raise RuntimeError("PLE fast reader is closed")
        result = _Stats()
        status = self._run(self._pointer, num_tokens, num_reqs, -1, C.byref(result))
        if status:
            raise RuntimeError(f"PLE fast residency probe failed (status={status})")
        return result

    def run(self, num_tokens: int, num_reqs: int, *, publish: bool = True) -> None:
        if not self._reader._finalize.alive:
            raise RuntimeError("PLE fast reader is closed")
        status = self._run(self._pointer, num_tokens, num_reqs, publish, self._stats_pointer)
        self._record(num_tokens, num_reqs)
        if status == 3:
            raise TimeoutError("Timed out waiting for PLE consumed flags")
        if status == 4:
            raise OSError("PLE fast row read failed or reached EOF; ready was not published")
        if status:
            raise RuntimeError(f"PLE fast lookup failed (status={status})")

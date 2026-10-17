# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only policy for the opt-in SM80 W8 experiment (changes numerics)."""

from dataclasses import dataclass
import os


# (scope, N, K, count per PP24 target stage). Control BA is deliberately absent.
INVENTORY = (
    ("gdn", 16384, 2560, 18),
    ("qsa", 13312, 2560, 6),
    ("out", 2560, 6144, 24),
    ("indexer", 640, 2560, 6),
)
SCOPES = frozenset(row[0] for row in INVENTORY)


@dataclass(frozen=True)
class Settings:
    mode: str
    scopes: frozenset[str]
    group_size: int = -1


def settings(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get("QWEN38_W8_DENSE", "").strip().lower()
    if mode in ("", "0", "off"):
        return None
    if mode not in ("fp8", "int8"):
        raise ValueError("QWEN38_W8_DENSE must be off, fp8 or int8")
    scopes = frozenset(s.strip() for s in env.get(
        "QWEN38_W8_SCOPE", "gdn,qsa,out,indexer").split(","))
    if not scopes or not scopes <= SCOPES:
        raise ValueError("QWEN38_W8_SCOPE must be a comma-separated subset of "
                         "gdn,qsa,out,indexer")
    group = env.get("QWEN38_W8_GROUP_SIZE", "-1")
    if group not in ("-1", "128"):
        raise ValueError("QWEN38_W8_GROUP_SIZE must be -1 (channel) or 128")
    return Settings(mode, scopes, int(group))


def projection_scope(name, shape):
    """Name AND physical shape allowlist; never enroll by shape alone."""
    if len(shape) != 2:
        return None
    paths = (
        (".linear_attn.in_proj_qkvz", "gdn", (16384, 2560)),
        (".self_attn.qkv_proj", "qsa", (13312, 2560)),
        (".linear_attn.out_proj", "out", (2560, 6144)),
        (".self_attn.o_proj", "out", (2560, 6144)),
        (".self_attn.indexer.index_qk_proj", "indexer", (640, 2560)),
    )
    for suffix, scope, expected in paths:
        if ("." + name).endswith(suffix) and tuple(shape) == expected:
            return scope
    return None


def stage_storage(group_size=-1, scopes=SCOPES, sms=0):
    """Persistent tensor bytes, not measured allocator/driver memory."""
    if group_size not in (-1, 128):
        raise ValueError("unsupported W8 group size")
    before = after = count = 0
    for scope, n, k, copies in INVENTORY:
        if scope not in scopes:
            continue
        before += copies * 2 * n * k
        after += copies * (n * k + 2 * n * (1 if group_size == -1 else k // 128)
                           + 4 * sms)
        count += copies
    return dict(layers=count, bf16_bytes=before, w8_bytes=after,
                freed_bytes=before - after)

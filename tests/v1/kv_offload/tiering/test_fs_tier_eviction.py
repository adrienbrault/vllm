# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
    FileSystemTierMetrics,
)

_BLOCK_SIZE = 4096
_GIB = 1024**3


def _key(value: int) -> OffloadKey:
    return make_offload_key(value.to_bytes(8, "big"), 0)


def _spec() -> MagicMock:
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="eviction-test",
        model=OffloadingModelConfig(name="eviction-test-model", dtype="uint8"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=True,
        ),
        replicated_layout=False,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=False,
        self_describing_kv_events=False,
    )
    return spec


@pytest.fixture
def make_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct", lambda _path: False
    )
    tiers: list[FileSystemTierManager] = []

    def create(**kwargs) -> FileSystemTierManager:
        data = np.arange(8 * _BLOCK_SIZE, dtype=np.uint8).reshape(8, _BLOCK_SIZE)
        tier = FileSystemTierManager(
            offloading_spec=_spec(),
            primary_kv_view=memoryview(data),
            tier_type="fs",
            root_dir=str(tmp_path),
            n_read_threads=1,
            n_write_threads=1,
            **kwargs,
        )
        tier._test_data = data
        tiers.append(tier)
        return tier

    yield create
    for tier in tiers:
        tier.shutdown()


def _job(
    job_id: int,
    keys: list[OffloadKey],
    *,
    load: bool = False,
    ctx: ReqContext | None = None,
) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=keys,
        block_ids=np.arange(len(keys), dtype=np.int64),
        is_promotion=load,
        req_context=ctx or ReqContext(req_id=f"job-{job_id}"),
    )


def _drain(tier: FileSystemTierManager):
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def _lookup(tier: FileSystemTierManager, key: OffloadKey, ctx: ReqContext):
    assert tier.lookup(key, ctx) is LookupResult.RETRY
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            return tier.lookup(key, ctx)
        time.sleep(0.01)
    pytest.fail("filesystem lookup did not finish")


def _capacity_for(blocks: int) -> float:
    return blocks * _BLOCK_SIZE / _GIB


def _store(tier: FileSystemTierManager, job_id: int, *values: int):
    keys = [_key(value) for value in values]
    tier.submit_store(_job(job_id, keys))
    results = _drain(tier)
    assert len(results) == 1 and results[0].success
    return keys


def test_capacity_is_respected_under_stream_of_stores(make_tier):
    tier = make_tier(
        max_capacity_gb=_capacity_for(2),
        evict_high_ratio=0.9,
        evict_low_ratio=0.5,
    )

    for value in range(5):
        _store(tier, value, value)
        assert tier._bytes_used <= 2 * _BLOCK_SIZE

    assert tier._bytes_used == _BLOCK_SIZE
    assert list(tier._lru) == [tier.file_mapper.get_file_name(_key(4))]
    stats = tier.get_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[FileSystemTierMetrics.EVICTED_FILES][()] == 4
    assert values[FileSystemTierMetrics.EVICTED_BYTES][()] == 4 * _BLOCK_SIZE


def test_factory_forwards_eviction_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct", lambda _path: False
    )
    data = np.zeros((2, _BLOCK_SIZE), dtype=np.uint8)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "max_capacity_gb": _capacity_for(2),
            "evict_high_ratio": 0.95,
            "evict_low_ratio": 0.75,
            "min_free_gb": _capacity_for(1),
            "evict_scope": "root",
        },
        memoryview(data),
        _spec(),
    )
    try:
        assert tier._max_capacity == 2 * _BLOCK_SIZE
        assert tier._high_watermark == int(2 * _BLOCK_SIZE * 0.95)
        assert tier._low_watermark == int(2 * _BLOCK_SIZE * 0.75)
        assert tier._min_free == _BLOCK_SIZE
        assert tier._scope_dir == str(tmp_path)
    finally:
        tier.shutdown()


def test_lookup_touch_controls_lru_order(make_tier):
    tier = make_tier(
        max_capacity_gb=_capacity_for(3),
        evict_high_ratio=1.0,
        evict_low_ratio=2 / 3,
    )
    a, b, c = _store(tier, 1, 1, 2, 3)
    ctx = ReqContext(req_id="touch-a")
    assert _lookup(tier, a, ctx) is LookupResult.HIT
    tier.on_request_finished(ctx)

    (d,) = _store(tier, 2, 4)

    assert list(tier._lru) == [
        tier.file_mapper.get_file_name(a),
        tier.file_mapper.get_file_name(d),
    ]
    assert not os.path.exists(tier.file_mapper.get_file_name(b))
    assert not os.path.exists(tier.file_mapper.get_file_name(c))


def test_pinned_hit_survives_and_store_is_skipped_when_full(make_tier):
    tier = make_tier(
        max_capacity_gb=_capacity_for(1),
        evict_high_ratio=1.0,
        evict_low_ratio=0.5,
    )
    (pinned,) = _store(tier, 1, 1)
    ctx = ReqContext(req_id="pin-owner")
    assert _lookup(tier, pinned, ctx) is LookupResult.HIT

    (skipped,) = _store(tier, 2, 2)

    assert os.path.exists(tier.file_mapper.get_file_name(pinned))
    assert not os.path.exists(tier.file_mapper.get_file_name(skipped))
    stats = tier.get_stats()
    assert stats is not None
    values = stats.data["data"]
    assert values[FileSystemTierMetrics.BYTES_USED][()] == _BLOCK_SIZE
    assert values[FileSystemTierMetrics.FILES][()] == 1
    assert values[FileSystemTierMetrics.SKIPPED_STORES][()] == 1
    tier.on_request_finished(ctx)
    assert tier._pin_counts == {}


def test_completed_load_touches_and_releases_lookup_and_job_pins(make_tier):
    tier = make_tier(max_capacity_gb=_capacity_for(2))
    (loaded,) = _store(tier, 1, 1)
    ctx = ReqContext(req_id="load-owner")
    assert _lookup(tier, loaded, ctx) is LookupResult.HIT
    path = tier.file_mapper.get_file_name(loaded)
    assert tier._pin_counts[path] == 1

    tier.submit_load(_job(2, [loaded], load=True, ctx=ctx))
    assert tier._pin_counts[path] == 2
    assert all(result.success for result in _drain(tier))

    assert tier._pin_counts == {}
    assert list(tier._lru)[-1] == path


def test_startup_accounts_prepopulated_namespace_by_access_time(make_tier):
    seed = make_tier()
    paths = [Path(seed.file_mapper.get_file_name(_key(i))) for i in (1, 2)]
    seed.shutdown()
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([i]) * _BLOCK_SIZE)
        timestamp = 100 + i
        os.utime(path, (timestamp, timestamp))

    tier = make_tier(max_capacity_gb=_capacity_for(3))

    assert tier._bytes_used == 2 * _BLOCK_SIZE
    assert list(tier._lru) == [str(path) for path in paths]


def test_eviction_tolerates_file_that_vanishes(make_tier):
    seed = make_tier()
    old_path = Path(seed.file_mapper.get_file_name(_key(1)))
    pinned_path = Path(seed.file_mapper.get_file_name(_key(2)))
    seed.shutdown()
    for path in (old_path, pinned_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * _BLOCK_SIZE)
    os.utime(old_path, (100, 100))
    os.utime(pinned_path, (200, 200))

    tier = make_tier(
        max_capacity_gb=_capacity_for(1),
        evict_high_ratio=0.9,
        evict_low_ratio=0.5,
    )
    ctx = ReqContext(req_id="pin-newer")
    tier.on_new_request(ctx)
    assert tier.batch_lookup_and_pin([str(pinned_path)], ctx) == [True]
    old_path.unlink()

    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    tier.drain_jobs()

    assert tier._bytes_used == _BLOCK_SIZE
    assert list(tier._lru) == [str(pinned_path)]
    tier.on_request_finished(ctx)


def test_min_free_triggers_background_eviction(make_tier, monkeypatch):
    free_bytes = [10 * _BLOCK_SIZE]
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.os.statvfs",
        lambda _path: SimpleNamespace(f_bavail=free_bytes[0], f_frsize=1),
    )
    tier = make_tier(min_free_gb=_capacity_for(1))
    (stored,) = _store(tier, 1, 1)
    path = tier.file_mapper.get_file_name(stored)
    assert os.path.exists(path)

    free_bytes[0] = 0
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    tier.drain_jobs()

    assert not os.path.exists(path)
    assert tier._bytes_used == 0


def test_root_scope_evicts_foreign_namespace_first(make_tier, tmp_path):
    seed = make_tier()
    own_path = Path(seed.file_mapper.get_file_name(_key(1)))
    seed.shutdown()
    foreign_path = tmp_path / "foreign-model_deadbeef_r0" / "000" / "00_g0" / "f.bin"
    for path, timestamp in ((foreign_path, 100), (own_path, 200)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * _BLOCK_SIZE)
        os.utime(path, (timestamp, timestamp))

    tier = make_tier(
        max_capacity_gb=_capacity_for(2),
        evict_high_ratio=0.75,
        evict_low_ratio=0.5,
        evict_scope="root",
    )
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    tier.drain_jobs()

    assert not foreign_path.exists()
    assert own_path.exists()
    assert tier._bytes_used == _BLOCK_SIZE

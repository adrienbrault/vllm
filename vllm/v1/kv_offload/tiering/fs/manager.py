# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FileSystemTierManager: pure-Python filesystem tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.tiering.metrics import FileSystemTierMetrics

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

_GIB = 1024**3
_EVICTION_BATCH_SIZE = 128


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        return self._tier.batch_lookup_and_pin(paths, req_context)


class FileSystemTierManager(SecondaryTierManager):
    """Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        KV cache sharing between multiple vLLM instances using the same
        ``root_dir`` (e.g., via a shared PVC) works by default: ``NONE_HASH``
        (the chain-hash seed for block content hashes) is derived from a fixed
        default seed, so identical token content produces identical block
        filenames across instances. Setting the ``PYTHONHASHSEED`` environment
        variable to the same value on all instances overrides the default seed,
        and is required to share a cache when using a non-cryptographic
        prefix-caching hash algorithm, which seeds ``NONE_HASH`` randomly.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        max_capacity_gb: float = 0,
        evict_high_ratio: float = 0.90,
        evict_low_ratio: float = 0.80,
        min_free_gb: float = 0,
        evict_scope: str = "namespace",
        backpressure_detector: BackpressureDetector | None = None,
    ):
        """Args:
        offloading_spec: Contains normalized offloading configuration and
            blocks_per_chunk.
        primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        tier_type: Tier type identifier, set by SecondaryTierFactory.
        root_dir: Root directory for block files.
        n_read_threads: Number of read-priority I/O threads.
        n_write_threads: Number of write-priority I/O threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.

        """
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, backpressure_detector
        )
        self.locality = Locality(locality) if locality is not None else None
        if max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be non-negative")
        if min_free_gb < 0:
            raise ValueError("min_free_gb must be non-negative")
        if not 0 <= evict_low_ratio < evict_high_ratio <= 1:
            raise ValueError(
                "eviction ratios must satisfy "
                "0 <= evict_low_ratio < evict_high_ratio <= 1"
            )
        if evict_scope not in ("namespace", "root"):
            raise ValueError("evict_scope must be 'namespace' or 'root'")
        self._max_capacity = int(max_capacity_gb * _GIB)
        self._high_watermark = int(self._max_capacity * evict_high_ratio)
        self._low_watermark = int(self._max_capacity * evict_low_ratio)
        self._min_free = int(min_free_gb * _GIB)
        self._eviction_enabled = self._max_capacity > 0 or self._min_free > 0


        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Store keys are needed only for optional events. Paths are always
        # tracked so pending jobs stay pinned until their result is collected.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        self._store_job_paths: dict[JobId, list[str]] = {}
        self._store_skipped_keys: dict[JobId, set[OffloadKey]] = {}
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        self._load_job_paths: dict[JobId, list[str]] = {}
        self._load_job_req_ids: dict[JobId, str] = {}
        # Block count per in-flight job, used to report transfer_bytes.
        self._job_block_counts: dict[JobId, int] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._scope_dir = (
            root_dir
            if evict_scope == "root"
            else f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
        )
        # Scheduler lookup/request methods and pool workers share this metadata.
        # Filesystem I/O is never performed by the scheduler; eviction holds the
        # lock for at most one small unlink batch before yielding it.
        self._cache_lock = threading.Lock()
        self._lru: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._bytes_used = 0
        self._reserved_bytes = 0
        self._reservations: set[str] = set()
        self._pin_counts: dict[str, int] = {}
        self._request_pins: dict[str, set[str]] = {}
        self._active_requests: set[str] = set()
        self._eviction_scheduled = False
        self._evicted_files = 0
        self._evicted_bytes = 0
        self._skipped_stores = 0
        self._scan_existing_files()

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    def _scan_existing_files(self) -> None:
        """Build initial accounting with one DirEntry.stat per block file."""
        records: list[tuple[int, str, int]] = []
        pending_dirs = [self._scope_dir]
        while pending_dirs:
            directory = pending_dirs.pop()
            try:
                entries = os.scandir(directory)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(
                    "Failed to scan fs tier directory %s: %s", directory, exc
                )
                continue
            with entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending_dirs.append(entry.path)
                        elif entry.name.endswith(".bin"):
                            info = entry.stat(follow_symlinks=False)
                            if stat.S_ISREG(info.st_mode):
                                last_access = max(info.st_atime_ns, info.st_mtime_ns)
                                records.append((last_access, entry.path, info.st_size))
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        logger.warning(
                            "Failed to inspect fs tier path %s: %s", entry.path, exc
                        )
        for last_access, path, size in sorted(records):
            self._lru[path] = (size, last_access)
            self._bytes_used += size

    def _forget_path_locked(self, path: str) -> None:
        record = self._lru.pop(path, None)
        if record is not None:
            self._bytes_used -= record[0]
            assert self._bytes_used >= 0

    def _record_path_locked(self, path: str, *, touch: bool = True) -> bool:
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            self._forget_path_locked(path)
            return False
        except OSError as exc:
            logger.warning("Failed to account fs tier path %s: %s", path, exc)
            return False
        if not stat.S_ISREG(info.st_mode):
            self._forget_path_locked(path)
            return False
        previous = self._lru.pop(path, None)
        if previous is not None:
            self._bytes_used -= previous[0]
        last_access = (
            time.time_ns() if touch else max(info.st_atime_ns, info.st_mtime_ns)
        )
        self._lru[path] = (info.st_size, last_access)
        self._bytes_used += info.st_size
        return True

    def _touch_path_locked(self, path: str) -> bool:
        record = self._lru.pop(path, None)
        if record is None:
            return self._record_path_locked(path)
        self._lru[path] = (record[0], time.time_ns())
        return True

    def _pin_paths_locked(self, paths: Collection[str]) -> None:
        for path in paths:
            self._pin_counts[path] = self._pin_counts.get(path, 0) + 1

    def _unpin_paths_locked(self, paths: Collection[str]) -> None:
        for path in paths:
            count = self._pin_counts[path] - 1
            if count:
                self._pin_counts[path] = count
            else:
                del self._pin_counts[path]

    def _pin_request_path_locked(self, req_id: str, path: str) -> None:
        if req_id not in self._active_requests:
            return
        paths = self._request_pins.setdefault(req_id, set())
        if path not in paths:
            paths.add(path)
            self._pin_paths_locked((path,))

    def _release_request_paths_locked(
        self, req_id: str, paths: Collection[str] | None = None
    ) -> None:
        request_paths = self._request_pins.get(req_id)
        if not request_paths:
            return
        releasing = (
            request_paths if paths is None else request_paths.intersection(paths)
        )
        self._unpin_paths_locked(releasing)
        request_paths.difference_update(releasing)
        if not request_paths:
            self._request_pins.pop(req_id, None)

    def batch_lookup_and_pin(
        self, paths: list[str], req_context: ReqContext
    ) -> list[bool]:
        """Probe and pin hits atomically with respect to internal eviction."""
        # Temporarily pin every candidate before the filesystem probe. This
        # closes the lookup-to-pin gap without holding _cache_lock across the
        # potentially slow batch call.
        with self._cache_lock:
            self._pin_paths_locked(paths)
        try:
            if _HAS_BATCH_LOOKUP_C:
                # C extension: GIL released for the faccessat() batch.
                hits = list(batch_lookup_C(paths))
            else:
                hits = [os.path.exists(path) for path in paths]
        except Exception:
            with self._cache_lock:
                self._unpin_paths_locked(paths)
            raise
        with self._cache_lock:
            for index, (path, hit) in enumerate(zip(paths, hits)):
                if not hit:
                    self._forget_path_locked(path)
                    continue
                if not self._touch_path_locked(path):
                    hits[index] = False
                    continue
                self._pin_request_path_locked(req_context.req_id, path)
            self._unpin_paths_locked(paths)
            return hits

    def _free_bytes_locked(self) -> int | None:
        try:
            fs_info = os.statvfs(self._scope_dir)
        except FileNotFoundError:
            fs_info = os.statvfs(os.path.dirname(self._scope_dir))
        except OSError as exc:
            logger.warning("Failed to read free space for %s: %s", self._scope_dir, exc)
            return None
        return fs_info.f_bavail * fs_info.f_frsize

    def _eviction_targets_locked(self, capacity_limit: int) -> tuple[int | None, int]:
        target_used = None
        if self._max_capacity:
            projected = self._bytes_used + self._reserved_bytes
            if projected > capacity_limit:
                target_used = max(0, self._low_watermark - self._reserved_bytes)
        free_needed = 0
        if self._min_free:
            free_bytes = self._free_bytes_locked()
            if free_bytes is not None:
                free_needed = max(0, self._min_free + self._reserved_bytes - free_bytes)
        return target_used, free_needed

    def _evict_to_limits(self, capacity_limit: int) -> None:
        """Delete unpinned LRU files in bounded lock-holding batches."""
        while True:
            with self._cache_lock:
                target_used, free_needed = self._eviction_targets_locked(capacity_limit)
                if target_used is None and free_needed == 0:
                    return
                changed = False
                bytes_freed = 0
                attempts = 0
                for path, (size, _last_access) in list(self._lru.items()):
                    if self._pin_counts.get(path, 0) or path in self._reservations:
                        continue
                    attempts += 1
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        self._forget_path_locked(path)
                        changed = True
                    except OSError as exc:
                        logger.warning("Failed to evict fs tier file %s: %s", path, exc)
                    else:
                        self._forget_path_locked(path)
                        self._evicted_files += 1
                        self._evicted_bytes += size
                        bytes_freed += size
                        changed = True
                    capacity_done = (
                        target_used is None or self._bytes_used <= target_used
                    )
                    if capacity_done and bytes_freed >= free_needed:
                        break
                    if attempts >= _EVICTION_BATCH_SIZE:
                        break
                if not changed:
                    return

    def _store_fits_locked(self) -> bool:
        if (
            self._max_capacity
            and self._bytes_used + self._reserved_bytes > self._max_capacity
        ):
            return False
        if self._min_free:
            free_bytes = self._free_bytes_locked()
            if free_bytes is None or free_bytes - self._reserved_bytes < self._min_free:
                return False
        return True

    def _prepare_store(
        self, job_id: JobId, keys: list[OffloadKey], paths: list[str]
    ) -> list[int]:
        selected: list[int] = []
        seen: set[str] = set()
        with self._cache_lock:
            for index, path in enumerate(paths):
                if path in seen:
                    continue
                seen.add(path)
                if self._record_path_locked(path):
                    continue
                if path in self._reservations:
                    continue
                self._reservations.add(path)
                self._reserved_bytes += self._block_size
                selected.append(index)
        if not selected:
            return selected

        self._evict_to_limits(self._max_capacity)
        with self._cache_lock:
            if self._store_fits_locked():
                return selected
            skipped = self._store_skipped_keys.setdefault(job_id, set())
            for index in selected:
                path = paths[index]
                self._reservations.remove(path)
                self._reserved_bytes -= self._block_size
                skipped.add(keys[index])
            self._skipped_stores += len(selected)
        return []

    def _store_task(
        self,
        job_id: JobId,
        keys: list[OffloadKey],
        paths: list[str],
        offsets: list[int],
    ) -> None:
        selected = self._prepare_store(job_id, keys, paths)
        try:
            if selected:
                batch_store_block(
                    [paths[index] for index in selected],
                    self._primary_kv_view,
                    [offsets[index] for index in selected],
                    self._block_size,
                    self._use_o_direct,
                )
        finally:
            with self._cache_lock:
                for index in selected:
                    path = paths[index]
                    self._reservations.remove(path)
                    self._reserved_bytes -= self._block_size
                    self._record_path_locked(path)
            self._evict_to_limits(self._high_watermark)

    def _schedule_eviction(self) -> None:
        if not self._eviction_enabled:
            return
        with self._cache_lock:
            if self._eviction_scheduled:
                return
            if self._min_free == 0 and self._bytes_used <= self._high_watermark:
                return
            self._eviction_scheduled = True
        self._pool.enqueue_store_task(self._run_scheduled_eviction)

    def _run_scheduled_eviction(self) -> None:
        try:
            self._evict_to_limits(self._high_watermark)
        finally:
            with self._cache_lock:
                self._eviction_scheduled = False

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        del extra_config
        return {
            FileSystemTierMetrics.BYTES_USED: OffloadingGaugeMetadata(
                documentation="Bytes currently accounted in the fs tier scope."
            ),
            FileSystemTierMetrics.FILES: OffloadingGaugeMetadata(
                documentation="Block files currently accounted in the fs tier scope."
            ),
            FileSystemTierMetrics.EVICTED_FILES: OffloadingCounterMetadata(
                documentation="Block files evicted from the fs tier."
            ),
            FileSystemTierMetrics.EVICTED_BYTES: OffloadingCounterMetadata(
                documentation="Bytes evicted from the fs tier."
            ),
            FileSystemTierMetrics.SKIPPED_STORES: OffloadingCounterMetadata(
                documentation="Block stores skipped because the fs tier was full."
            ),
        }

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        with self._cache_lock:
            self._active_requests.add(req_context.req_id)
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        with self._cache_lock:
            self._active_requests.add(req_context.req_id)
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        if not result:
            return LookupResult.MISS
        path = self.file_mapper.get_file_name(key)
        with self._cache_lock:
            if self._touch_path_locked(path):
                self._pin_request_path_locked(req_context.req_id, path)
                return LookupResult.HIT
        self._lookup_manager.mark_miss((key,))
        return LookupResult.MISS

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(cid) * self._block_size for cid in job_metadata.chunk_ids]
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        self._store_job_paths[job_metadata.job_id] = paths
        with self._cache_lock:
            self._pin_paths_locked(paths)
        task = functools.partial(
            self._store_task,
            job_metadata.job_id,
            keys,
            paths,
            offsets,
        )
        self._job_block_counts[job_metadata.job_id] = len(keys)
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        self._job_block_counts[job_id] = len(keys)
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        self._load_job_paths[job_id] = paths
        self._load_job_req_ids[job_id] = job_metadata.req_context.req_id
        with self._cache_lock:
            self._pin_paths_locked(paths)
        offsets = [int(cid) * self._block_size for cid in job_metadata.chunk_ids]

        def load_task() -> None:
            try:
                batch_load_block(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
                with self._cache_lock:
                    for path in paths:
                        self._touch_path_locked(path)
            except OSError as exc:
                # Runs on the pool worker thread. Record how many blocks loaded
                # before the failure so get_finished_jobs can keep them; this
                # write precedes task_done, so the scheduler reads it safely
                # under the GIL once the finished queue hands back this job.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                self._load_progress[job_id] = num_succeeded
                with self._cache_lock:
                    for path in paths[:num_succeeded]:
                        self._touch_path_locked(path)
                    for path in paths[num_succeeded:]:
                        if not os.path.exists(path):
                            self._forget_path_locked(path)
                # Surfaces errno (e.g. EMFILE "Too many open files") for both
                # the C and Python load paths.
                logger.debug(
                    "Load of %d blocks for job %s failed at block %d: %s",
                    len(paths),
                    job_id,
                    num_succeeded,
                    exc,
                )
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
            block_count = self._job_block_counts.pop(job_id, 0)
            transfer_bytes = block_count * self._block_size if block_count else None
            store_paths = self._store_job_paths.pop(job_id, None)
            with self._cache_lock:
                skipped_keys = self._store_skipped_keys.pop(job_id, set())
                if store_paths is not None:
                    self._unpin_paths_locked(store_paths)
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    event_keys = [key for key in keys if key not in skipped_keys]
                else:
                    event_keys = []
                if event_keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=event_keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            load_keys = self._load_job_keys.pop(job_id, None)
            load_paths = self._load_job_paths.pop(job_id, None)
            load_req_id = self._load_job_req_ids.pop(job_id, None)
            if load_paths is not None:
                with self._cache_lock:
                    self._unpin_paths_locked(load_paths)
                    assert load_req_id is not None
                    self._release_request_paths_locked(load_req_id, load_paths)
            num_succeeded = self._load_progress.pop(job_id, 0)
            if load_keys is not None and not success:
                # A batched load stops at the first bad block and reports how
                # many loaded before it. Those earlier blocks are kept in the
                # primary tier (reported via successful_keys); only this block
                # and the ones after it are marked a miss and recomputed.
                successful = load_keys[:num_succeeded]
                failed = load_keys[num_succeeded:]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                        transfer_bytes=transfer_bytes,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                    transfer_bytes=transfer_bytes,
                )
            )
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        with self._cache_lock:
            self._active_requests.discard(req_context.req_id)
            self._release_request_paths_locked(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()
        self._schedule_eviction()

    @override
    def get_stats(self) -> OffloadingConnectorStats:
        stats = OffloadingConnectorStats()
        with self._cache_lock:
            stats.set_gauge(FileSystemTierMetrics.BYTES_USED, self._bytes_used)
            stats.set_gauge(FileSystemTierMetrics.FILES, len(self._lru))
            if self._evicted_files:
                stats.increase_counter(
                    FileSystemTierMetrics.EVICTED_FILES, self._evicted_files
                )
            if self._evicted_bytes:
                stats.increase_counter(
                    FileSystemTierMetrics.EVICTED_BYTES, self._evicted_bytes
                )
            if self._skipped_stores:
                stats.increase_counter(
                    FileSystemTierMetrics.SKIPPED_STORES, self._skipped_stores
                )
            self._evicted_files = 0
            self._evicted_bytes = 0
            self._skipped_stores = 0
        return stats

    @override
    def shutdown(self) -> None:
        """Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)

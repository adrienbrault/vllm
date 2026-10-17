// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Plain C ABI: no CUDA or ATen. Scratch space and sleeping I/O workers are
// created at registration, never per request.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <thread>
#include <mutex>
#include <vector>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>

struct PleFastBinding {
  int64_t ngram, heads, dim, shard_rows, shards_count, targets_count;
  int64_t max_tokens, max_reqs, eos, clamp_inputs, timeout_ns;
  const int32_t *input, *starts, *context;
  const int64_t *multipliers, *sizes, *offsets;
  const uint8_t *const *shards;
  const int64_t *rows;
  uint8_t *const *outputs;
  int32_t *const *ready;
  int32_t *const *consumed;
  const int32_t *fds;       // null for an in-RAM table
  const int64_t *bases;     // file offsets of the shard views
  void *reader;
};

struct PleFastStats {
  int64_t rows = 0, nonresident = 0, unknown = 0, nowait_hits = 0;
  int64_t deferred = 0, wall_ns = 0, wait_ns = 0, read_ns = 0;
};

using Clock = std::chrono::steady_clock;
static int64_t elapsed(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct Row {
  const uint8_t *source;
  uint8_t *dest;
  int fd;
  int64_t offset;
};

// mincore observes page-cache residency without faulting the row into this
// process. Count the entire row, including an unaligned crossing of pages.
// An error is "unknown", never a cache hit. Residency is only diagnostic:
// mmap memcpy is NOT used for file-backed hits (reclaim can race mincore).
static int resident(const uint8_t *source, int64_t size, int64_t page) {
  uintptr_t first = reinterpret_cast<uintptr_t>(source) / page * page;
  const uintptr_t end = reinterpret_cast<uintptr_t>(source) + size;
  bool missing = false;
  while (first < end) {
    const size_t pages = std::min<uintptr_t>(64, (end - first + page - 1) / page);
#ifdef __APPLE__
    char vec[64];
#else
    unsigned char vec[64];
#endif
    if (mincore(reinterpret_cast<void *>(first), pages * page, vec) != 0) return -1;
    for (size_t i = 0; i < pages; ++i) missing |= !(vec[i] & 1);
    first += pages * page;
  }
  return missing ? 0 : 1;
}

class Reader {
 public:
  Reader(int64_t capacity, int threads)
      : rows(capacity), pending(capacity), page(sysconf(_SC_PAGESIZE)) {
    try {
      for (int i = 0; i < threads; ++i) workers.emplace_back([this] { work(); });
    } catch (...) {
      stop();
      throw;
    }
  }
  ~Reader() { stop(); }
  std::vector<Row> rows;
  std::vector<int64_t> pending;
  const int64_t page;
  bool nowait = true;  // unsupported filesystems stay on the pool thereafter

  int read(int64_t count, int64_t bytes, PleFastStats &stats) {
    const auto start = Clock::now();
    int64_t misses = 0;
    // Snapshot every row before any read can bring neighboring rows into cache.
    for (int64_t i = 0; i < count; ++i) {
      const Row &r = rows[i];
      if (r.fd < 0) continue;
      const int hit = resident(r.source, bytes, page);
      stats.nonresident += hit == 0;
      stats.unknown += hit < 0;
    }
    for (int64_t i = 0; i < count; ++i) {
      const Row &r = rows[i];
      if (r.fd < 0) {  // non-mmap, original contiguous in-RAM FP8 table
        std::memcpy(r.dest, r.source, bytes);
        continue;
      }
#if defined(__linux__) && defined(RWF_NOWAIT)
      if (nowait) {
        iovec vector{r.dest, static_cast<size_t>(bytes)};
        const ssize_t done = preadv2(r.fd, &vector, 1, r.offset, RWF_NOWAIT);
        if (done == bytes) {
          ++stats.nowait_hits;
          continue;
        }
        if (done < 0 && (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS))
          nowait = false;
        // EAGAIN, EINTR, a partial/zero read or another error: workers retry
        // the WHOLE row; no partially initialized bytes can be published.
      }
#else
      nowait = false;
#endif
      pending[misses++] = i;
    }
    stats.deferred = misses;
    if (misses) {
      std::unique_lock<std::mutex> lock(mutex);
      length = bytes;
      pending_count = misses;
      cursor.store(0, std::memory_order_relaxed);
      error.store(0, std::memory_order_relaxed);
      finished = 0;
      ++generation;
      available.notify_all();
      complete.wait(lock, [this] { return finished == workers.size(); });
    }
    stats.read_ns = elapsed(start);
    return misses ? error.load(std::memory_order_relaxed) : 0;
  }

 private:
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable available, complete;
  bool stopping = false;
  uint64_t generation = 0;
  size_t finished = 0;
  int64_t pending_count = 0, length = 0;
  std::atomic<int64_t> cursor{0};
  std::atomic<int> error{0};

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      available.notify_all();
    }
    for (auto &thread : workers) if (thread.joinable()) thread.join();
  }
  void work() {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
      available.wait(lock, [&] { return stopping || generation != seen; });
      if (stopping) return;
      seen = generation;
      lock.unlock();
      for (;;) {
        const int64_t index = cursor.fetch_add(1, std::memory_order_relaxed);
        if (index >= pending_count) break;
        const Row &r = rows[pending[index]];
        int64_t done = 0;
        while (done < length) {
          const ssize_t n = pread(r.fd, r.dest + done, length - done, r.offset + done);
          if (n < 0 && errno == EINTR) continue;
          if (n <= 0) {
            error.store(4, std::memory_order_relaxed);
            break;
          }
          done += n;
        }
      }
      lock.lock();
      if (++finished == workers.size()) complete.notify_one();
    }
  }
};

extern "C" int ple_fast_abi_version() { return 2; }
extern "C" void *ple_fast_create(int64_t capacity, int threads) {
  if (capacity <= 0 || threads <= 0 || threads > 8) return nullptr;
  try { return new Reader(capacity, threads); } catch (...) { return nullptr; }
}
// Caller must have drained run() before closing this registration.
extern "C" void ple_fast_destroy(void *reader) { delete static_cast<Reader *>(reader); }

struct Measure {
  PleFastStats &stats;
  Clock::time_point start = Clock::now();
  ~Measure() { stats.wall_ns = elapsed(start); }
};

// Return codes: 0 success, 1 invalid batch, 2 invalid table row, 3 wait timeout.
// 4 read error/EOF (no ready is published).
// publish=0: lookup only; publish=-1: plan/probe residency without reading.
// These entries let CPU tests measure eviction without warming rows.
extern "C" int ple_fast_run(const PleFastBinding *b, int64_t tokens,
                            int64_t reqs, int publish, PleFastStats *result) {
  PleFastStats ignored;
  PleFastStats &stats = result ? *result : ignored;
  stats = PleFastStats{};
  Measure measure{stats};
  auto *reader = static_cast<Reader *>(b->reader);
  if (!reader) return 1;
  if (tokens < 0 || tokens > 64 || tokens > b->max_tokens ||
      reqs <= 0 || reqs > b->max_reqs || b->starts[0] != 0)
    return 1;
  int64_t width = 1;
  for (int64_t r = 0; r < reqs; ++r) {
    const int64_t length = int64_t(b->starts[r + 1]) - b->starts[r];
    if (length < 0) return 1;
    width = std::max(width, length);
  }
  // The eager path's packed buffer has max_total_tokens columns.
  if (width > b->max_tokens) return 1;
  const int64_t valid = std::min<int64_t>(b->starts[reqs], tokens);
  const int64_t context_len = b->ngram - 1;
  const int64_t total_heads = context_len * b->heads;
  const int64_t row_bytes = total_heads * b->dim;
  if (tokens * total_heads > static_cast<int64_t>(reader->rows.size())) return 1;
  stats.rows = tokens * total_heads;

  if (publish > 0) {
    // Preserve the existing GPU -> CPU ownership transfer. Do not assume a
    // new request means all TP copies of the previous result have completed.
    const auto wait_start = Clock::now();
    const auto deadline = wait_start +
                          std::chrono::nanoseconds(b->timeout_ns);
    for (int64_t t = 0; t < b->targets_count; ++t) {
      while (__atomic_load_n(b->consumed[t], __ATOMIC_ACQUIRE) != 1) {
        if (Clock::now() >= deadline) {
          stats.wait_ns = elapsed(wait_start);
          return 3;
        }
        // CUDA writes do not futex-wake CPU waiters. A timed sleep gives up
        // the core without requiring any change to the captured GPU graph.
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    stats.wait_ns = elapsed(wait_start);
    for (int64_t t = 0; t < b->targets_count; ++t) {
      __atomic_store_n(b->ready[t], 0, __ATOMIC_RELEASE);
      __atomic_store_n(b->consumed[t], 0, __ATOMIC_RELEASE);
    }
  }

  // Read the eager path's virtual [context | EOS-filled packed] tensor.
  // Padding tokens are deliberately mapped with its same searchsorted/clamp
  // rule, including repeating the last real token when the last row is full.
  for (int64_t t = 0; t < tokens; ++t) {
    int64_t r = std::upper_bound(b->starts, b->starts + reqs + 1, t) -
                b->starts - 1;
    r = std::min(r, reqs - 1);
    const int64_t col = std::clamp<int64_t>(t - b->starts[r], 0, width - 1);
    const int64_t pos = col + context_len;
    auto token_at = [&](int64_t p) -> int64_t {
      if (p < 0) return b->eos;
      if (p < context_len) return b->context[r * context_len + p];
      const int64_t c = p - context_len;
      const int64_t i = int64_t(b->starts[r]) + c;
      if (i >= valid || i >= b->starts[r + 1]) return b->eos;
      const int64_t value = b->input[i];
      return b->clamp_inputs ? std::max<int64_t>(0, value) : value;
    };
    // Unsigned multiplication gives precisely torch.int64 wraparound, without
    // signed-overflow UB. EOS resets affect shifted history, not the current ID.
    uint64_t mixed = uint64_t(token_at(pos)) * uint64_t(b->multipliers[0]);
    bool boundary = false;
    for (int64_t shift = 1; shift < b->ngram; ++shift) {
      int64_t previous = token_at(pos - shift);
      boundary = boundary || previous == b->eos;
      if (boundary) previous = b->eos;
      mixed ^= uint64_t(previous) * uint64_t(b->multipliers[shift]);
      int64_t signed_mixed;
      std::memcpy(&signed_mixed, &mixed, sizeof(mixed));
      for (int64_t h = 0; h < b->heads; ++h) {
        const int64_t head = (shift - 1) * b->heads + h;
        int64_t rem = signed_mixed % b->sizes[head];
        if (rem < 0) rem += b->sizes[head];
        const int64_t id = rem + b->offsets[head];
        const int64_t shard = id / b->shard_rows;
        const int64_t local = id % b->shard_rows;
        if (id < 0 || shard >= b->shards_count || local >= b->rows[shard])
          return 2;
        reader->rows[t * total_heads + head] = Row{
            b->shards[shard] + local * b->dim,
            b->outputs[0] + t * row_bytes + head * b->dim,
            b->fds ? b->fds[shard] : -1,
            b->bases ? b->bases[shard] + local * b->dim : 0};
      }
    }
  }
  if (publish < 0) {
    for (int64_t i = 0; i < stats.rows; ++i) {
      const Row &row = reader->rows[i];
      if (row.fd < 0) continue;
      const int hit = resident(row.source, b->dim, reader->page);
      stats.nonresident += hit == 0;
      stats.unknown += hit < 0;
    }
    return 0;
  }
  const int status = reader->read(stats.rows, b->dim, stats);
  if (status) return status;
  if (publish > 0) {
    for (int64_t t = 1; t < b->targets_count; ++t)
      std::memcpy(b->outputs[t], b->outputs[0], tokens * row_bytes);
    // All output stores and TP fanout finish before any ready publication.
    for (int64_t t = 0; t < b->targets_count; ++t)
      __atomic_store_n(b->ready[t], 1, __ATOMIC_RELEASE);
  }
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Plain C ABI: no CUDA, ATen dispatch, worker threads, or request allocations.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

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
};

static inline void relax_cpu() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

// Return codes: 0 success, 1 invalid batch, 2 invalid table row, 3 wait timeout.
// publish=0 is a lookup-only entry for CPU differential tests.
extern "C" int ple_fast_run(const PleFastBinding *b, int64_t tokens,
                            int64_t reqs, int publish) {
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

  if (publish) {
    // Preserve the existing GPU -> CPU ownership transfer. Do not assume a
    // new request means all TP copies of the previous result have completed.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds(b->timeout_ns);
    for (int64_t t = 0; t < b->targets_count; ++t) {
      while (__atomic_load_n(b->consumed[t], __ATOMIC_ACQUIRE) != 1) {
        for (int spin = 0; spin < 64; ++spin) relax_cpu();
        if (std::chrono::steady_clock::now() >= deadline) return 3;
        std::this_thread::yield();
      }
    }
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
        std::memcpy(b->outputs[0] + t * row_bytes + head * b->dim,
                    b->shards[shard] + local * b->dim, b->dim);
      }
    }
  }
  if (publish) {
    for (int64_t t = 1; t < b->targets_count; ++t)
      std::memcpy(b->outputs[t], b->outputs[0], tokens * row_bytes);
    // All output stores and TP fanout finish before any ready publication.
    for (int64_t t = 0; t < b->targets_count; ++t)
      __atomic_store_n(b->ready[t], 1, __ATOMIC_RELEASE);
  }
  return 0;
}

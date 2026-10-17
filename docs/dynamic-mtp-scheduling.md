# Opt-in dynamic MTP CUDA graphs

`QWEN38_DYNAMIC_MTP_GRAPHS=1` enables the Qwen3.8 MTP3/MTP2 scheduling fix.
It defaults to off. It requires an explicit
`num_speculative_tokens_per_batch_size` schedule, maximum draft depth 3,
scheduled depths 2 or 3, async scheduling, and DP=1. The tested source target
is the V2 runner with PP=2 and FULL_DECODE_ONLY; GPU validation is pending.

For three drafts with up to two running requests and two drafts above that:

```bash
export QWEN38_DYNAMIC_MTP_GRAPHS=1
EXTRA_VLLM_ARGS='--mamba-ssm-cache-dtype float32 --speculative-config {"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"model":"/model/runtime/mtp-int4-g32","num_speculative_tokens_per_batch_size":[[1,2,3],[3,16,2]]}'
```

Retain `QWEN38_PP=2`, `QWEN38_ASYNC_SCHEDULING=1`, `MAX_NUM_SEQS=16`, and
the existing FULL_DECODE_ONLY compilation config. The docker launcher forwards
exported `QWEN38_*` settings. `serve-container.sh` already enables the dynamic
scheduler machinery. The new flag uses **all running requests** as the lookup
index, including resident prefill requests; the old path uses requests in the
current PP microbatch. Budget changes apply to the next proposals and can lag
the current verification by the pipeline's in-flight steps.

The MTP draft's internal config revalidation now has its own compilation and
speculative config. It cannot downgrade PP1's target config or clear its graph
sizes. KV cache geometry and attention registration remain shared. Target and
draft-prefill graphs cover query lengths 4 and 3; single-token draft decode has
query length 1. Dispatch searches compatible larger graph shapes when the
nearest token count only has the other query length. Each shape retains the
padding of its fixed-depth baseline. There are no new numeric kernels.

The original fused MTP3 graph remains in use at c1. Fused MTP2 decode has its
own graph; non-fused attention backends replay their single-step graph fewer
times instead. Proposals and PP buffers retain maximum width 3, sampled output
transport retains width 4, and the scheduler consumes only the selected K
columns. No PP queue, eligibility cadence, placeholder accounting, rollback
rule, or target sampling rule is changed.

At INFO level, worker-prefixed `[dynamic-mtp] captured` lines report actual
captured `(tokens, padded requests, query length)` triples. Both workers must
report target shapes for query 3 and 4, and PP1 must capture the speculator.
`[dynamic-mtp] replay` reports the first actual replay of each graph shape;
`[dynamic-mtp] draft next_k=2` and `next_k=3` report proposal-depth selection.
Use `VLLM_LOGGING_LEVEL=DEBUG` for a correctness run to log every `replay-step`,
`draft-step`, and eager dispatch. Prefill and mixed-transition batches may run
eagerly. Steady uniform decode at c1/c4/c8 must use FULL on both stages.

CPU regression:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_dynamic_mtp_graphs.py
```

Before adoption, compare all generated token IDs under temperature 0 against
fixed MTP3 and fixed MTP2, at c1/c4/c8 and during 3-to-2-to-3 transitions.
Include aborts/slot reuse, chunked prefill, repeated prefixes, and structured
outputs. Require zero token mismatches, shape disagreement, deadlocks, or
preemptions, then measure at least three warmed screens without DEBUG logging.
No speedup or full-model greedy parity has been measured for this patch.

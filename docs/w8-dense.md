# Experimental SM80 W8 dense projections

`QWEN38_W8_DENSE=fp8` or `int8` quantizes selected target BF16 weights once
after checkpoint loading, then uses the installed Marlin W8A16 kernel for all
row counts, including prefill. It changes target numerics. The default is off;
this source change has no GPU performance or model-quality approval.

`QWEN38_W8_SCOPE=gdn,qsa,out,indexer` is the default selection. Use a subset
to stage quality and performance tests. This is a name-and-shape allowlist for
the PP24/TP1 target:

| Scope | Physical weight (N,K) | Count per target stage |
|---|---|---:|
| gdn | (16384,2560), fused QKVZ | 18 |
| qsa | (13312,2560), fused QKV/output-gate | 6 |
| out | (2560,6144), GDN/QSA output | 24 |
| indexer | (640,2560), replicated Q/K | 6 |

The gates already fused inside QKVZ/QSA matrices share their quantization.
Separate GDN BA/control, HC, routers, shared experts, PLE, vocabulary heads,
vision and the MTP draft remain under their original methods. Changing TP
geometry is outside this allowlist.

Scales are symmetric per output channel by default. Set
`QWEN38_W8_GROUP_SIZE=128` only as a separately measured arm. INT8 uses biased
`uint8b128`, FP8 uses E4M3FN; both retain BF16 activations and FP32 reductions.
No activation quantization or new CUDA kernels are added. No BF16 fallback
weight is retained for prefill. The normal LinearBase wrappers still own TP
reductions, bias and output tuple behavior.

Conversion is enrolled before weight loading and runs through the normal
per-layer quantization finalize hook. The BF16 Parameter is deleted after
successful repacking. `W8_DENSE` logs per-layer bytes and `W8_DENSE_TOTAL` logs
the model/rank total, including scale and workspace buffers. Allocator reserve
can remain cached and is reusable by KV allocations. With all four scopes,
channel scales save about 1.2534 GiB/stage, group-128 about 1.2347 GiB/stage
(74-SM illustration; the kernel queries actual SM count).

Only CUDA SM80 resident BF16 weights are supported; offloaded selected weights
and unsupported active settings fail explicitly. Checkpoint refit/reload,
sleep/wake and saving the runtime packed buffers as a checkpoint are outside
this experiment: restart from the original BF16 checkpoint for each arm.

CPU checks: `python3 -B tests/test_w8_dense_cpu.py`. The task's accompanying
`out/bench_w8.py`, `out/test_w8.py`, `out/quality_w8.py` and `out/REPORT.md`
contain the GPU screen, layer/graph checks and operator sequence. Benchmark
winners do not alter the runtime policy automatically. Keep the same KV
allocation for matched speed tests; evaluate extra KV capacity in a separate
run after quality passes.

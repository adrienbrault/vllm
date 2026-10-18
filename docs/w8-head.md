# Experimental SM80 INT8 vocabulary heads

Add `head` and/or `draft_head` to `QWEN38_W8_SCOPE` with
`QWEN38_W8_DENSE=int8`. Both scopes default off. Existing `gdn,qsa,out`
deployments retain their original behavior. Channel scales use
`QWEN38_W8_GROUP_SIZE=-1`; `128` is a separate numerical arm. FP8 head
scopes are rejected. Quantization changes weight numerics.

The serving loader replaces the MTP head with the **same module object** as
the target head after both models finish loading. Caller scope therefore
cannot be inferred from the head object alone. Target projection uses the
head method's `apply`; MTP's `DraftLogitsProcessor` explicitly uses
`apply_draft` for full logits, prefill and unrestricted greedy sampling.
Restricted MTP sampling uses that method's `draft_packed` selector.

| Explicit head scopes | Target projection | Draft projection | Resident shared head |
|---|---|---|---|
| none | Original BF16 | Original BF16 | One BF16 weight |
| `head` | Full-vocabulary W8 | Original BF16 | BF16 plus one W8 weight |
| `draft_head` | Original BF16 | W8, reduced when configured | BF16 plus one segmented W8 weight |
| `head,draft_head` | Full-vocabulary W8 | W8, reduced when configured | One segmented W8 weight |

Single-scope runs retain the original BF16 Parameter for the unselected
caller. It shares the original allocation, without cloning. BF16 full-head
calls retain the existing skinny/linear method selection; restricted calls
retain the original BF16 range projection. Selecting both scopes releases
the BF16 Parameter and lets both callers reuse the same packed segments.
A target without a drafter conservatively retains BF16 in a single-scope run
as well. The MTP backbone remains under its existing policy.

When `draft_head` is selected, the **target's eventual shared head** is packed
at the boundaries from `VLLM_MTP_DRAFT_VOCAB_RANGES`, before the loader can
discard the drafter's separately loaded head. The served ranges produce
`[0,65536)`, `[65536,248044)`, `[248044,248320)` segments. Complement rows
preserve full-vocabulary `compute_logits`, profile and unrestricted sampling.
Reduced drafting evaluates only the two requested segments. Full target logits
in the combined arm concatenate three GEMMs; remeasure this path in serving.
MTP3 reuses these weights three times, without three resident heads.

Marlin requires N divisible by 64: the 276-row tail pads to 320 and the
182,508-row complement to 182,528. Padding is trimmed before soft cap, scale,
max or token mapping. Any other valid requested interval uses overlapping
packed segments and slices their outputs. Full profile ranges, subranges and
cross-boundary ranges are correct without repacking, changing persistent
state, device-to-host reads or host synchronization. Only empty/out-of-bounds
intervals are rejected. An unfamiliar range can compute extra rows; its
fallback is for correctness, not a promised speedup.

Outputs remain BF16. The pinned logits processor retains vocabulary slicing,
bias, soft cap, scale and sampler behavior. Restricted sampling retains the
lowest-ID tie within sorted ranges and first-rank tie across TP ranks.
Supported W8 heads are resident SM80 BF16 `(248320,2560)` TP1
`ParallelLMHead` modules with BF16 or unspecified `head_dtype`. Tying to the
input embedding remains unsupported; sharing between target and draft is
supported. LoRA, weight reload/refit, sleep/wake and packed checkpoint export
remain outside this experiment. Restart from the original checkpoint per arm.

For actual SM count S (queried by the kernel), one BF16 head is
1,271,398,400 bytes. Define full W8 T = 636,195,840 + 4S bytes for channel
scales, or 645,632,000 + 4S for group-128. Define segmented W8 D =
636,359,808 + 12S bytes for channel, or 645,798,400 + 12S for group-128.

| Shared serving arm | Total head bytes | Freed bytes versus one shared BF16 head |
|---|---:|---:|
| `head` | 1,271,398,400 + T | −T (about 0.592/0.601 GiB added, channel/g128) |
| `draft_head` | 1,271,398,400 + D | −D (about 0.593/0.601 GiB added) |
| `head,draft_head` | D | 1,271,398,400 − D (about 0.591/0.583 GiB freed) |

These changes affect only the last PP rank. GPU0 gains zero head memory.
Do not sum savings for two heads on a module the loader shares. `W8_HEAD`
and `W8_DENSE_TOTAL` include retained BF16 and allow negative `freed_bytes`.
Enrollment reports both projection and head-module counts: stage 1 with
`gdn,qsa,out,head` has 49 modules (48 projections, 1 head module); stage 0 has
48 projections and no head. Draft-constructor logs describe its temporary
head before sharing. Do not sum target and draft pre-binding memory totals:
the draft's old head is discarded. Startup peaks, allocator reserve, graph
buffers and NVML memory still require operator measurements. Keep the served
21-GiB/card KV pool fixed for comparisons.

Run CPU checks with `python3 -B tests/test_w8_head_cpu.py` or
`out/test_w8head.py --cpu`. The serving-image regression
`out/test_w8head.py --shared-only --device 1` exercises independent enrollment
and finalization, loader sharing, full profile calls followed by reduced and
unconfigured ranges, both callers' full logits/prefill, and changed-input
CUDA graph replay for each scope combination and both group sizes. The full
script also retains packing bounds, numerical/argmax statistics, bias,
empty/3D inputs and tie checks. `out/REPORT.md` gives operator commands.
Previous isolated GPU screens passed, but the corrected shared serving path
still needs GPU boot, capture, memory, throughput, acceptance and quality
validation. Synthetic agreement does not establish model quality.

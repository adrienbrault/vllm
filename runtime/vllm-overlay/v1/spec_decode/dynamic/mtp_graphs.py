# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in graph support for the Qwen3.8 MTP3/MTP2 batch-size schedule."""

import os


def dynamic_mtp_graphs_enabled(vllm_config) -> bool:
    spec = vllm_config.speculative_config
    return (
        os.environ.get("QWEN38_DYNAMIC_MTP_GRAPHS", "0") == "1"
        and spec is not None
        and spec.method == "mtp"
        and spec.uses_dynamic_speculative_decoding()
    )


def dynamic_mtp_batch_size(vllm_config, num_scheduled: int, num_running: int) -> int:
    # PP splits running requests into eligible microbatches. Select the NEXT
    # proposal budget by resident concurrency, not half that concurrency.
    return num_running if dynamic_mtp_graphs_enabled(vllm_config) else num_scheduled

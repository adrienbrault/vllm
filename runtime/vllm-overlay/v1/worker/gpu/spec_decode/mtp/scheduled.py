# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Variable draft work with fixed-width request state and PP transports."""

from contextlib import contextmanager

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

logger = init_logger(__name__)


class ScheduledMTPSpeculator(MTPSpeculator):
    def __init__(self, vllm_config, device):
        spec = vllm_config.speculative_config
        if (
            spec.num_speculative_tokens != 3
            or any(k not in (2, 3) for _, _, k in spec.num_speculative_tokens_per_batch_size)
            or vllm_config.parallel_config.data_parallel_size != 1
            or not vllm_config.scheduler_config.async_scheduling
        ):
            raise ValueError(
                "QWEN38_DYNAMIC_MTP_GRAPHS requires MTP3/MTP2, DP=1, and async scheduling"
            )
        super().__init__(vllm_config, device)
        self.num_draft_tokens = self.num_speculative_steps
        self.mtp2_decode_cudagraph_manager = None
        self._logged_depths = set()

    def init_cudagraph_manager(self, cudagraph_mode):
        super().init_cudagraph_manager(cudagraph_mode)
        if self.use_fused_multi_step_decode and any(
            k == 2 for _, _, k in
            self.vllm_config.speculative_config.num_speculative_tokens_per_batch_size
        ):
            # Fused graphs bake the remaining Python loop into the graph. Keep
            # the original K=3 graph for c1 and capture a separate K=2 version.
            self.mtp2_decode_cudagraph_manager = SpeculatorCudaGraphManager(
                self.vllm_config, self.device,
                self.decode_cudagraph_manager.cudagraph_mode, decode_query_len=1,
            )

    def capture(self):
        super().capture()
        if self.mtp2_decode_cudagraph_manager is not None:
            self.on_multi_step_decode_begin(self.max_num_reqs)
            try:
                with self._draft_decode_context(2):
                    self.decode_cudagraph_manager.capture(
                        self._generate_fused_drafts, self.model_state,
                        self.input_buffers, self.block_tables, self.attn_groups,
                        self.kv_cache_config,
                        progress_bar_desc="Capturing MTP2 decode CUDA graphs",
                    )
            finally:
                self.on_multi_step_decode_end(self.max_num_reqs)

    @contextmanager
    def _draft_decode_context(self, depth=None):
        depth = self.num_draft_tokens if depth is None else depth
        max_steps = self.num_speculative_steps
        max_manager = self.decode_cudagraph_manager
        if depth not in (2, 3):
            raise ValueError(f"Unexpected scheduled MTP depth: {depth}")
        # Only bound the loop; buffers, prefill metadata, sampler strides, and
        # returned proposals retain their maximum width of 3. The scheduler
        # never reads the unused trailing column when it selected K=2.
        if self.use_fused_multi_step_decode and depth == 2:
            assert self.mtp2_decode_cudagraph_manager is not None
        self.num_speculative_steps = depth
        if self.use_fused_multi_step_decode and depth == 2:
            self.decode_cudagraph_manager = self.mtp2_decode_cudagraph_manager
        try:
            yield
        finally:
            self.num_speculative_steps = max_steps
            self.decode_cudagraph_manager = max_manager

    def _log_depth(self):
        logger.debug("[dynamic-mtp] draft-step next_k=%d fused=%s state_width=3",
                     self.num_draft_tokens, self.use_fused_multi_step_decode)
        if self.num_draft_tokens not in self._logged_depths:
            self._logged_depths.add(self.num_draft_tokens)
            logger.info("[dynamic-mtp] draft next_k=%d fused=%s state_width=3",
                        self.num_draft_tokens, self.use_fused_multi_step_decode)

    def _multi_step_decode(self, *args, **kwargs):
        self._log_depth()
        with self._draft_decode_context():
            return super()._multi_step_decode(*args, **kwargs)

    def _fused_multi_step_decode(self, *args, **kwargs):
        self._log_depth()
        with self._draft_decode_context():
            return super()._fused_multi_step_decode(*args, **kwargs)

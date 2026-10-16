# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP2 PCIe all-reduce for the R185 served image.

No exception-driven fallback after launching a collective. Initialization
failures are agreed across the CPU group; per-call gates are metadata-only.
"""

import json
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger

logger = init_logger(__name__)


class PcieIpcAllReduce:
    def __init__(self, group, device, custom_comm, allowed):
        self.disabled = True
        self.workspace = None
        self.config = get_current_vllm_config_or_none()
        self.extra_shapes = []
        self.hiddens = {5120}
        reason = None
        # Static group facts: never enter TP2 ownership code at another size.
        if dist.get_world_size(group) != 2:
            reason = "requires world size 2"
        elif dist.get_backend(group) != "gloo":
            reason = "requires the existing TP Gloo control group"
        if reason is not None:
            logger.info("PCIe IPC all-reduce disabled: %s", reason)
            return

        # Keep this first exchange independent of the vendored package: a
        # missing/broken installation on one rank cannot strand its peer.
        error = None
        try:
            reason = self._disabled_reason(allowed, custom_comm, device)
            from pcie_ipc_ar21.workspace import (
                JointFailure,
                PcieIpcAllReduceWorkspace,
            )
            if reason is None:
                self.extra_shapes = self._parse_shapes(
                    os.environ.get("VLLM_SM12X_PCIE_IPC_PREPARE_SHAPES", "[]")
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        with torch.inference_mode(False):
            outcomes = [None, None]
            dist.all_gather_object(outcomes, (reason, error, self.extra_shapes), group=group)
        if any(item[0] or item[1] for item in outcomes) or outcomes[0] != outcomes[1]:
            logger.info("PCIe IPC all-reduce disabled: rank prerequisites %s", outcomes)
            return

        try:
            self.workspace = PcieIpcAllReduceWorkspace(group, device)
            self.prepare_for_capture(self.config)
        except JointFailure as exc:
            if self.workspace is not None:
                self.workspace.destroy()
                self.workspace = None
            logger.info("PCIe IPC all-reduce disabled: %s", exc)
            return
        # Set only after all allocations, agreements and preparation succeeded.
        self.disabled = False
        logger.info(
            "PCIe IPC all-reduce enabled: TP2 bf16, max_rows=320, "
            "max_numel=1638400, slab_bytes=26223744; fixed R184 tactics "
            "transplanted from max_numel=41943040 (no boot autotuning)"
        )

        if (self.config.speculative_config is not None
                and self.config.speculative_config.method == "mtp"):
            from pcie_ipc_ar21.fixed_config import MAX_ROWS

            logger.info_once(
                "SM12X PCIe IPC: MTP drafter admitted "
                "(sequential AutoRegressive drafter, rows<=%d)", MAX_ROWS,
                scope="process",
            )

    def _disabled_reason(self, allowed, custom_comm, device):
        if not allowed:
            return "custom all-reduce disabled, non-TP group, or non-CUDA platform"
        if custom_comm is None or custom_comm.disabled or custom_comm.mnnvl_only:
            return "requires enabled same-node CustomAllreduce/P2P fallback"
        if torch.cuda.get_device_capability(device) != (12, 0):
            return "requires sm_120"
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5090":
            return "fixed tuning data requires RTX 5090"
        if torch.version.cuda != "13.0":
            return "fixed tuning data requires CUDA 13.0 torch build"
        config = self.config
        if config is None:
            return "worker config unavailable at communicator construction"
        parallel = config.parallel_config
        if parallel.enable_dbo:
            return "DBO may overlap forward streams"
        if any(getattr(parallel, name, 1) != 1 for name in (
            "pipeline_parallel_size", "data_parallel_size",
            "prefill_context_parallel_size", "decode_context_parallel_size",
        )):
            return "only TP2 with PP/DP/PCP/DCP=1 is supported"
        model = config.model_config
        if model.dtype != torch.bfloat16 or model.get_hidden_size() != 5120:
            return "requires bf16 target hidden size 5120"
        if model.is_moe or config.lora_config is not None:
            return "MoE/LoRA auxiliary forward streams are outside this contract"
        if model.enable_sleep_mode:
            return "raw IPC allocations do not support sleep/offload lifecycle"
        spec = config.speculative_config
        mtp_requested = envs.VLLM_SM12X_PCIE_IPC_MTP
        if spec is not None and spec.method != "dflash":
            if not (mtp_requested and spec.method == "mtp"):
                return "only the sequential DFlash drafter is supported"
        if mtp_requested:
            if spec is None or spec.method != "mtp":
                return "VLLM_SM12X_PCIE_IPC_MTP=1 requires speculative method mtp"
            passes = config.compilation_config.pass_config
            if passes.enable_sp or passes.fuse_gemm_comms or passes.fuse_allreduce_rms:
                return "PCIe IPC MTP requires unfused tensor-parallel all-reduce dispatch"
            draft = spec.draft_model_config
            if (draft.dtype != torch.bfloat16 or draft.get_hidden_size() != 5120
                    or spec.draft_parallel_config.tensor_parallel_size != 2):
                return "PCIe IPC MTP requires a bf16 hidden=5120 TP2 drafter"
        return None

    @staticmethod
    def _parse_shapes(text):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("PREPARE_SHAPES must be a JSON list of [rows, hidden]")
        for shape in value:
            if not (isinstance(shape, list) and len(shape) == 2
                    and all(type(x) is int and x > 0 for x in shape)):
                raise ValueError("PREPARE_SHAPES entries must be two positive integers")
        return sorted(set(tuple(shape) for shape in value))

    def _prepare(self, shapes, dtype=torch.bfloat16):
        if self.workspace is None:
            return
        # The complete request is agreed even on repeat calls; skipping a
        # collective based on rank-local cache state can deadlock the group.
        added = self.workspace.prepare(shapes, dtype)
        self.hiddens.update(h for _, h in added)
        if added:
            logger.info(
                "PCIe IPC prepared resolved shapes=%s dtype=%s; hidden=5120 "
                "uses fixed floor buckets, other hiddens use the TP2 seed",
                sorted(added), dtype,
            )

    def prepare_for_capture(self, config=None):
        from pcie_ipc_ar21.fixed_config import CAPTURE_ROWS, MAX_ROWS

        from pcie_ipc_ar21.workspace import stage

        config = config or self.config

        def capture_shapes():
            sizes = (config.compilation_config.cudagraph_capture_sizes
                     if config is not None else None) or CAPTURE_ROWS
            shapes = [(int(rows), 5120) for rows in sizes] + self.extra_shapes
            # Cover post-config padding/adjustment too, even for runner helpers
            # absent from the dump. This is CPU metadata, not 320 GPU buffers.
            shapes += [(rows, 5120) for rows in range(1, MAX_ROWS + 1)]
            spec = config.speculative_config if config is not None else None
            if spec is not None and spec.method == "dflash":
                hidden = spec.draft_model_config.get_hidden_size()
                # The dumped CudaGraphManager base/config is missing. Resolve a
                # bounded superset of EVERY possible padded query row count, not
                # an invented num_reqs * query_len capture schedule.
                shapes += [(rows, hidden) for rows in range(1, MAX_ROWS + 1)]
            return shapes

        if self.workspace is not None:
            shapes = stage(self.workspace.group, "reading capture shapes", capture_shapes)
            self._prepare(shapes)

    def prepare_draft(self, hidden, dtype):
        from pcie_ipc_ar21.fixed_config import MAX_ROWS

        self._prepare([(rows, int(hidden)) for rows in range(1, MAX_ROWS + 1)], dtype)

    def should_use(self, inp):
        ws = self.workspace
        if self.disabled or ws is None or not ws.supports(inp):
            return False
        if inp.shape[1] not in self.hiddens:
            return False
        # No lazy resolution, collective, or exception for a capture cache miss.
        if torch.cuda.is_current_stream_capturing() and not ws.is_resolved(inp):
            return False
        return True

    def all_reduce(self, inp):
        return self.workspace.all_reduce(inp)

    def capture(self):
        return self.workspace.capture() if not self.disabled else nullcontext()

    def destroy(self):
        if self.workspace is not None:
            self.workspace.destroy()
            self.workspace = None
        self.disabled = True

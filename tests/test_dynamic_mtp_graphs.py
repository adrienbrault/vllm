"""CPU regression checks for dynamic MTP graph planning and PP shape contracts.

Execute the production methods with CUDA-free objects, like test_ple_capture_hook.
This does not establish full-model greedy parity or CUDA capture correctness.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from enum import Enum
from itertools import product
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "runtime/vllm-overlay"
GPU = OVERLAY / "v1/worker/gpu"


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2
    FULL_DECODE_ONLY = 3

    def __bool__(self):
        return self != Mode.NONE

    def decode_mode(self):
        return Mode.FULL if self else Mode.NONE

    def mixed_mode(self):
        return Mode.NONE

    def separate_routine(self):
        return True


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def enabled(config):
    return (os.environ.get("QWEN38_DYNAMIC_MTP_GRAPHS", "0") == "1"
            and config.speculative_config is not None
            and config.speculative_config.method == "mtp"
            and config.speculative_config.uses_dynamic_speculative_decoding())


def graph_manager(query_len=4, schedule=None, sizes=None, flag=True):
    namespace = dict(dataclass=dataclass, CUDAGraphMode=Mode, defaultdict=defaultdict,
                     product=product, round_up=lambda x, y: (x + y - 1) // y * y,
                     dynamic_mtp_graphs_enabled=enabled,
                     logger=NS(debug=lambda *args: None))
    tree = ast.parse((GPU / "cudagraph_utils.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "CudaGraphManager")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ("_init_candidates", "needs_capture", "dispatch",
                               "_resolve_effective_loras")]
    desc = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "BatchExecutionDescriptor")
    compat = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_is_compatible")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), desc, compat, cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(GPU / "cudagraph_utils.py"), "exec"), namespace)
    spec = NS(method="mtp", num_speculative_tokens_per_batch_size=schedule,
              uses_dynamic_speculative_decoding=lambda: schedule is not None)
    config = NS(speculative_config=spec, num_speculative_tokens=3)
    manager = namespace["CudaGraphManager"]()
    manager.__dict__.update(
        vllm_config=config, compilation_config=NS(
            cudagraph_capture_sizes=sizes or [1, 2, 4, *range(8, 129, 8)],
            max_cudagraph_capture_size=128),
        max_num_reqs=16, decode_query_len=query_len, varlen_decode=False,
        cudagraph_mode=Mode.FULL_DECODE_ONLY, lora_capture_cases=[0],
        _candidates={}, _capture_descs={}, _lora_dispatch_map={}, _max_lora_case=0,
        _graphs_captured=True, dynamic_mtp_graphs=flag and schedule is not None,
    )
    with patch.dict(os.environ, {"QWEN38_DYNAMIC_MTP_GRAPHS": "1" if flag else "0"}):
        manager._init_candidates()
    return manager


SCHEDULE = [[1, 2, 3], [3, 16, 2]]


class DynamicMTPGraphTests(unittest.TestCase):
    def test_both_verify_shapes_match_fixed_padding_for_every_c1_to_c8_batch(self):
        manager = graph_manager(schedule=SCHEDULE)
        self.assertTrue(manager.needs_capture())
        for qlen in (3, 4):
            fixed = graph_manager(query_len=qlen, flag=False)
            for reqs in range(1, 9):
                desc = manager.dispatch(reqs, reqs * qlen, qlen, 0)
                self.assertEqual(desc.cg_mode, Mode.FULL)
                self.assertEqual(vars(desc), vars(fixed.dispatch(reqs, reqs * qlen, qlen, 0)))

    def test_single_token_draft_decode_does_not_expand_target_schedule(self):
        manager = graph_manager(query_len=1, schedule=SCHEDULE)
        self.assertTrue(manager.needs_capture())
        for reqs in range(1, 9):
            self.assertEqual(manager.dispatch(reqs, reqs, 1, 0).cg_mode, Mode.FULL)

    def test_dispatch_searches_past_incompatible_nearest_size(self):
        # 6 tokens -> nearest captured size 8 (qlen=4), but qlen=3 needs size 9.
        manager = graph_manager(schedule=SCHEDULE, sizes=[8, 16])
        self.assertEqual(manager.dispatch(2, 6, 3, 0).cg_mode, Mode.FULL)

    def test_flag_off_preserves_original_planning(self):
        manager = graph_manager(schedule=SCHEDULE, flag=False)
        descs = manager._capture_descs[Mode.FULL]
        self.assertEqual(len(descs), 17)
        self.assertEqual(manager.dispatch(4, 12, 3, 0).cg_mode, Mode.NONE)
        self.assertEqual(len(graph_manager(flag=False)._capture_descs[Mode.FULL]), 9)

    def test_prefills_and_mixed_transition_batches_remain_eager(self):
        manager = graph_manager(schedule=SCHEDULE)
        for reqs, tokens, uniform in ((1, 20, 20), (2, 7, None), (2, 20, None)):
            self.assertEqual(manager.dispatch(reqs, tokens, uniform, 0).cg_mode, Mode.NONE)

    def test_draft_revalidation_cannot_mutate_target_graph_config(self):
        path = OVERLAY / "models/qwen3_8_flash_next/nvidia/mtp.py"
        def replace(config, **kwargs):
            draft = NS(**(vars(config) | kwargs))
            # VllmConfig post-init: dynamic SD + non-V2 draft -> PIECEWISE;
            # compilation NONE then disables CG and clears its capture sizes.
            if draft.speculative_config.num_speculative_tokens_per_batch_size:
                draft.compilation_config.cudagraph_mode = Mode.NONE
                draft.compilation_config.cudagraph_capture_sizes = []
                draft.compilation_config.max_cudagraph_capture_size = 0
            return draft
        namespace = load_nodes(path, {"_make_draft_vllm_config"}, dict(
            replace=replace, copy=copy, dynamic_mtp_graphs_enabled=enabled,
            get_draft_quant_config=lambda config: None,
        ))
        for flag in (False, True):
            spec = NS(method="mtp", draft_model_config=object(),
                      num_speculative_tokens_per_batch_size=SCHEDULE,
                      uses_dynamic_speculative_decoding=lambda: True)
            ctx = {}
            target = NS(speculative_config=spec, cache_config=NS(mamba_block_size=1600),
                        compilation_config=NS(cudagraph_mode=Mode.FULL_DECODE_ONLY,
                            cudagraph_capture_sizes=[1, 2, 4, 8],
                            max_cudagraph_capture_size=8, static_forward_context=ctx))
            with patch.dict(os.environ, {"QWEN38_DYNAMIC_MTP_GRAPHS": "1" if flag else "0"}):
                draft = namespace["_make_draft_vllm_config"](target, 48)
            self.assertEqual(target.compilation_config.cudagraph_mode,
                             Mode.FULL_DECODE_ONLY if flag else Mode.NONE)
            self.assertIs(draft.compilation_config.static_forward_context, ctx)
            self.assertIs(draft.cache_config, target.cache_config)
            self.assertEqual(target.cache_config.mamba_block_size, 1600)
            self.assertIs(spec.num_speculative_tokens_per_batch_size, SCHEDULE)

    def test_flag_defaults_off_and_schedule_uses_running_pp_concurrency(self):
        ns = {}
        path = OVERLAY / "v1/spec_decode/dynamic/mtp_graphs.py"
        exec(compile(path.read_text(), str(path), "exec"), ns)
        config = NS(speculative_config=NS(method="mtp",
                    uses_dynamic_speculative_decoding=lambda: True))
        for flag in ("0", "1"):
            with patch.dict(os.environ, {"QWEN38_DYNAMIC_MTP_GRAPHS": flag}):
                batch_size = ns["dynamic_mtp_batch_size"](config, 2, 4)
                self.assertEqual(batch_size, 4 if flag == "1" else 2)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(ns["dynamic_mtp_graphs_enabled"](config))

    def test_active_draft_loop_restores_max_width_even_on_error(self):
        calls = []

        class Base:
            def _multi_step_decode(self, fail=False):
                calls.extend(range(1, self.num_speculative_steps))
                if fail:
                    raise RuntimeError("test")

        ns = load_nodes(GPU / "spec_decode/mtp/scheduled.py",
                        {"ScheduledMTPSpeculator"}, {"MTPSpeculator": Base,
                         "contextmanager": contextmanager,
                         "logger": NS(info=lambda *args: None, debug=lambda *args: None)})
        drafter = object.__new__(ns["ScheduledMTPSpeculator"])
        drafter.num_speculative_steps = 3
        drafter.use_fused_multi_step_decode = False
        drafter.decode_cudagraph_manager = object()
        drafter._logged_depths = set()
        # Include both transitions; each proposal's prefill produces token 0.
        for k in (3, 2, 3, 2):
            calls.clear()
            drafter.num_draft_tokens = k
            drafter._multi_step_decode()
            self.assertEqual(calls, list(range(1, k)))
            self.assertEqual(drafter.num_speculative_steps, 3)
        with self.assertRaises(RuntimeError):
            drafter._multi_step_decode(fail=True)
        self.assertEqual(drafter.num_speculative_steps, 3)

    def test_fused_mtp3_keeps_original_graph_and_mtp2_selects_own_graph(self):
        calls = []
        class Base:
            def _fused_multi_step_decode(self):
                calls.append((self.num_speculative_steps, self.decode_cudagraph_manager))
        ns = load_nodes(GPU / "spec_decode/mtp/scheduled.py",
                        {"ScheduledMTPSpeculator"}, {"MTPSpeculator": Base,
                         "contextmanager": contextmanager,
                         "logger": NS(info=lambda *args: None, debug=lambda *args: None)})
        drafter = object.__new__(ns["ScheduledMTPSpeculator"])
        max_manager, mtp2_manager = object(), object()
        vars(drafter).update(num_speculative_steps=3, use_fused_multi_step_decode=True,
            decode_cudagraph_manager=max_manager, mtp2_decode_cudagraph_manager=mtp2_manager,
            _logged_depths=set())
        for k in (3, 2, 3):
            drafter.num_draft_tokens = k
            drafter._fused_multi_step_decode()
            self.assertEqual(calls[-1], (k, max_manager if k == 3 else mtp2_manager))
            self.assertEqual(drafter.num_speculative_steps, 3)
            self.assertIs(drafter.decode_cudagraph_manager, max_manager)

    def test_fused_mtp2_capture_bakes_two_steps_then_restores_mtp3(self):
        events = []
        class Base:
            def capture(self):
                events.append(("base", self.num_speculative_steps))
            def on_multi_step_decode_begin(self, n):
                events.append(("begin", n))
            def on_multi_step_decode_end(self, n):
                events.append(("end", n))
            def _generate_fused_drafts(self):
                pass
        ns = load_nodes(GPU / "spec_decode/mtp/scheduled.py",
                        {"ScheduledMTPSpeculator"}, {"MTPSpeculator": Base,
                         "contextmanager": contextmanager})
        drafter = object.__new__(ns["ScheduledMTPSpeculator"])
        max_manager = object()
        def capture(*args, **kwargs):
            events.append(("capture", drafter.num_speculative_steps))
            self.assertIs(drafter.decode_cudagraph_manager, mtp2_manager)
        mtp2_manager = NS(capture=capture)
        vars(drafter).update(num_speculative_steps=3, use_fused_multi_step_decode=True,
            decode_cudagraph_manager=max_manager, mtp2_decode_cudagraph_manager=mtp2_manager,
            max_num_reqs=16, model_state=None, input_buffers=None, block_tables=None,
            attn_groups=[], kv_cache_config=None)
        drafter.capture()
        self.assertEqual(events, [("base", 3), ("begin", 16), ("capture", 2), ("end", 16)])
        self.assertEqual(drafter.num_speculative_steps, 3)
        self.assertIs(drafter.decode_cudagraph_manager, max_manager)

    def test_budget_is_carried_by_execute_sample_pair_and_warmup_uses_max(self):
        tree = ast.parse((GPU / "model_runner.py").read_text())
        ctor = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "ExecuteModelState")
        expr = next(k.value for k in ctor.keywords if k.arg == "num_spec_tokens_to_schedule")
        code = compile(ast.Expression(expr), "<budget>", "eval")
        for k in (0, 2, 3):
            self.assertEqual(eval(code, {"self": NS(num_speculative_steps=3),
                             "scheduler_output": NS(num_spec_tokens_to_schedule=k)}), k or 3)

    def test_pp_stages_plan_identical_shapes_during_depth_transitions(self):
        # The draft loader may change only its local config. Both target
        # managers have max query 4, and current verification K can lag next K.
        stages = [graph_manager(schedule=SCHEDULE) for _ in range(2)]
        for n, current_k in ((1, 3), (2, 3), (2, 2), (4, 2), (4, 3), (1, 3)):
            descs = [m.dispatch(n, n * (current_k + 1), current_k + 1, 0)
                     for m in stages]
            self.assertEqual(vars(descs[0]), vars(descs[1]))
            self.assertEqual(descs[0].cg_mode, Mode.FULL)


if __name__ == "__main__":
    unittest.main()

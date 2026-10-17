"""GPU-free policy and real-method lifecycle tests; no torch installation needed."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia"
spec = importlib.util.spec_from_file_location("w8_test_policy", SOURCE / "w8_dense_policy.py")
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class W8PolicyTests(unittest.TestCase):
    def test_off_is_default_and_ignores_inactive_scope(self):
        self.assertIsNone(policy.settings({}))
        self.assertIsNone(policy.settings({"QWEN38_W8_DENSE": "off", "QWEN38_W8_SCOPE": "typo"}))

    def test_invalid_active_settings_fail(self):
        for env in ({"QWEN38_W8_DENSE": "1"},
                    {"QWEN38_W8_DENSE": "int8", "QWEN38_W8_SCOPE": "router"},
                    {"QWEN38_W8_DENSE": "int8", "QWEN38_W8_SCOPE": ""},
                    {"QWEN38_W8_DENSE": "fp8", "QWEN38_W8_GROUP_SIZE": "32"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                policy.settings(env)

    def test_default_scope_excludes_controls(self):
        config = policy.settings({"QWEN38_W8_DENSE": "int8"})
        self.assertEqual(config.scopes, frozenset(("gdn", "qsa", "out", "indexer")))
        self.assertEqual(config.group_size, -1)
        for name, shape, expected in (
            ("model.layers.0.linear_attn.in_proj_qkvz", (16384,2560), "gdn"),
            ("model.layers.3.self_attn.qkv_proj", (13312,2560), "qsa"),
            ("model.layers.0.linear_attn.out_proj", (2560,6144), "out"),
            ("model.layers.3.self_attn.o_proj", (2560,6144), "out"),
            ("model.layers.3.self_attn.indexer.index_qk_proj", (640,2560), "indexer"),
            ("model.layers.0.linear_attn.in_proj_ba", (96,2560), None),
            ("model.layers.0.mlp.gate", (16384,2560), None),
            ("model.layers.0.mlp.shared_expert.gate_up_proj", (1280,2560), None),
            ("model.layers.0.hyper_connection.up_proj", (2560,6144), None),
            ("model.layers.0.linear_attn.in_proj_qkvz", (4096,2560), None),
            ("lm_head", (248320,2560), None),
        ):
            with self.subTest(name=name, shape=shape):
                self.assertEqual(policy.projection_scope(name, shape), expected)

    def test_inventory_bytes_and_scope_union(self):
        channel, group = (policy.stage_storage(g) for g in (-1, 128))
        self.assertEqual(channel["layers"], 54)
        self.assertEqual(channel["bf16_bytes"], 18*16384*2560*2 + 6*13312*2560*2
                         + 24*2560*6144*2 + 6*640*2560*2)
        self.assertGreater(channel["freed_bytes"], group["freed_bytes"])
        parts = [policy.stage_storage(scopes={scope}, sms=74) for scope in policy.SCOPES]
        for key in channel:
            self.assertEqual(sum(part[key] for part in parts), policy.stage_storage(sms=74)[key])

    def test_real_lifecycle_and_enrollment_without_tensor_math(self):
        # Execute the actual class/enroll bodies with storage stand-ins. This
        # tests ownership/release and method dispatch, not CUDA/kernel numerics.
        dtype = object()

        class Weight:
            shape = (16384, 2560)
            def numel(self): return 100
            def element_size(self): return 2
        Weight.dtype = dtype

        class Unquantized:
            pass

        class Linear:
            def __init__(self):
                self.weight = Weight()
                self.quant_method = Unquantized()
            def add_module(self, name, obj): setattr(self, name, obj)

        class Packed:
            storage_bytes = 112
            def __init__(self, weight, mode, group): pass
            def __call__(self, x, bias): return (x, bias)

        layer = Linear()
        sensitive = Linear()
        root = SimpleNamespace(named_modules=lambda: [
            ("model.layers.0.linear_attn.in_proj_qkvz", layer),
            ("model.layers.0.linear_attn.in_proj_ba", sensitive)])
        namespace = dict(UnquantizedLinearMethod=Unquantized, PackedW8=Packed,
            LinearBase=Linear, torch=SimpleNamespace(bfloat16=dtype,
                cuda=SimpleNamespace(get_device_capability=lambda: (8, 0))),
            projection_scope=policy.projection_scope,
            settings=lambda: policy.settings({"QWEN38_W8_DENSE": "int8"}),
            log=SimpleNamespace(warning=lambda *a: None))
        tree = ast.parse((SOURCE / "w8_dense.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                     and n.name in ("Qwen38W8LinearMethod", "enroll")]
        exec(compile(tree, str(SOURCE / "w8_dense.py"), "exec"), namespace)
        namespace["enroll"](root, dtype)
        self.assertIs(type(sensitive.quant_method), Unquantized)
        method = layer.quant_method
        self.assertIsInstance(method, namespace["Qwen38W8LinearMethod"])
        method.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "weight"), "BF16 Parameter must be removed")
        packed = layer._qwen38_w8
        method.process_weights_after_loading(layer)
        self.assertIs(packed, layer._qwen38_w8)
        for rows in (4,8,16,32,2048):
            self.assertEqual(method.apply(layer, rows, "bias"), (rows, "bias"))
        draft = type("Qwen3_8FlashNextMTP", (), {})()
        namespace["enroll"](draft, dtype)  # no traversal/device work on MTP
        namespace["settings"] = lambda: None
        namespace["enroll"](object(), object())  # disabled path touches nothing

    def test_constructor_hook_runs_before_skinny(self):
        tree = ast.parse((SOURCE / "low_latency_gemm.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "enable_qwen38next_low_latency_gemm")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "enroll"]
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0].lineno, next(n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.For)))


if __name__ == "__main__":
    unittest.main()

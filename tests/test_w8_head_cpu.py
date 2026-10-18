"""GPU-free W8 head policy, enrollment guards and storage ownership tests."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
from test_w8_dense_cpu import policy, SOURCE


def definitions(filename, names, namespace):
    tree = ast.parse((SOURCE/filename).read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(tree, str(SOURCE/filename), "exec"), namespace)


class W8HeadTests(unittest.TestCase):
    def test_explicit_int8_opt_in(self):
        self.assertFalse(policy.settings({"QWEN38_W8_DENSE": "int8"}).scopes & policy.HEAD_SCOPES)
        for scope in ("head", "draft_head", "head,draft_head", "gdn,qsa,out,head,draft_head"):
            config = policy.settings({"QWEN38_W8_DENSE": "int8", "QWEN38_W8_SCOPE": scope})
            self.assertEqual(config.scopes, frozenset(scope.split(",")))
            with self.assertRaises(ValueError):
                policy.settings({"QWEN38_W8_DENSE": "fp8", "QWEN38_W8_SCOPE": scope})

    def test_partition_and_bytes(self):
        self.assertEqual(policy.head_segments(248320, policy.DRAFT_RANGES),
                         ((0,65536), (65536,248044), (248044,248320)))
        for group in (-1,128):
            target = policy.head_storage(group, sms=17)
            draft = policy.head_storage(group, sms=17, ranges=policy.DRAFT_RANGES)
            self.assertEqual(target["bf16_bytes"], 1271398400)
            self.assertEqual(target["padded_rows"], 248320)
            self.assertEqual(draft["padded_rows"], 248384)
            self.assertEqual(draft["w8_bytes"] - target["w8_bytes"],
                             64*(2560 + 2*(1 if group == -1 else 20)) + 2*4*17)
            self.assertGreater(draft["freed_bytes"], 600000000)
            shared_single = policy.head_storage(group,sms=17,ranges=policy.DRAFT_RANGES,retain_bf16=True)
            self.assertEqual(shared_single["retained_bf16_bytes"],1271398400)
            self.assertEqual(shared_single["freed_bytes"],-draft["w8_bytes"])
            self.assertEqual(shared_single["total_bytes"],1271398400+draft["w8_bytes"])
        self.assertEqual(policy.head_segments(100, ((0,100),)), ((0,100),))
        self.assertEqual(policy.head_segments(100, ((5,10),(10,15))),
                         ((0,5),(5,10),(10,15),(15,100)))

    def namespace(self):
        dtype = object()
        class Unquantized:
            def create_weights(self): pass
            def apply(self, layer, x, bias=None): return ("bf16", x, bias)
        class QuantBase: pass
        class Head:
            tp_size = 1
            def __init__(self):
                self.weight = NS(shape=policy.HEAD_SHAPE, dtype=dtype,
                                 numel=lambda: 100, element_size=lambda: 2)
                self.quant_method = Unquantized()
            def add_module(self, name, obj): setattr(self, name, obj)
        class Packed:
            storage_bytes = 112
            def __init__(self, weight, group, ranges): self.ranges = ranges
            def __call__(self, x, bias): return x, bias
        namespace = dict(QuantizeMethodBase=QuantBase, UnquantizedEmbeddingMethod=Unquantized,
                         ParallelLMHead=Head, PackedW8Head=Packed, HEAD_SHAPE=policy.HEAD_SHAPE,
                         HEAD_SCOPES=policy.HEAD_SCOPES, os=NS(environ={}),
                         torch=NS(bfloat16=dtype, cuda=NS(get_device_capability=lambda: (8,0))),
                         log=NS(warning=lambda *a: None))
        definitions("w8_head.py", {"Qwen38W8HeadMethod", "enroll_heads"}, namespace)
        return namespace, dtype, Head, Unquantized

    def model(self, head, dtype, draft=False):
        cls = type("Qwen3_8FlashNextMTP" if draft else "Target", (), {})
        root = cls()
        root.lm_head = head
        root.config = NS(tie_word_embeddings=False)
        root.logits_processor = NS(head_dtype=None)
        root._draft_vocab_ranges = policy.DRAFT_RANGES if draft else None
        return root

    def test_real_enrollment_lifecycle_target_and_draft(self):
        for draft, scope in ((False,"head"),(True,"draft_head")):
            ns, dtype, Head, Unquantized = self.namespace()
            root = self.model(Head(), dtype, draft)
            settings = policy.Settings("int8", frozenset({scope}))
            ns["enroll_heads"](root, dtype, settings)
            method = root.lm_head.quant_method
            self.assertIsInstance(method, ns["Qwen38W8HeadMethod"])
            self.assertEqual(method.ranges, policy.DRAFT_RANGES if draft else ())
            method.process_weights_after_loading(root.lm_head)
            self.assertEqual(hasattr(root.lm_head, "weight"), not draft)
            packed = root.lm_head._qwen38_w8_head
            method.process_weights_after_loading(root.lm_head)
            self.assertIs(root.lm_head._qwen38_w8_head, packed)
            for rows in (0,1,4,8,16,32,2048):
                self.assertEqual(method.apply(root.lm_head, rows, "bias"), (rows,"bias"))

    def test_stage_and_scope_isolation(self):
        ns, dtype, Head, Unquantized = self.namespace()
        target, draft = (self.model(Head(), dtype, d) for d in (False,True))
        ns["enroll_heads"](target, dtype, policy.Settings("int8", frozenset({"draft_head"})))
        ns["enroll_heads"](draft, dtype, policy.Settings("int8", frozenset({"head"})))
        self.assertIsInstance(target.lm_head.quant_method, ns["Qwen38W8HeadMethod"])
        self.assertEqual(target.lm_head.quant_method.apply(target.lm_head,"x"), ("bf16","x",None))
        self.assertIs(type(draft.lm_head.quant_method), Unquantized)
        target.lm_head = NS()  # stage 0: no head, no CUDA query
        ns["torch"].cuda.get_device_capability = lambda: self.fail("device touched on headless stage")
        ns["enroll_heads"](target, dtype, policy.Settings("int8", frozenset({"head"})))

    def test_shared_head_keeps_bf16_for_unselected_draft(self):
        ns, dtype, Head, Unquantized = self.namespace()
        target = self.model(Head(), dtype)
        ns["enroll_heads"](target, dtype, policy.Settings("int8", frozenset({"head"})))
        method = target.lm_head.quant_method
        weight = target.lm_head.weight
        method.process_weights_after_loading(target.lm_head)
        draft = self.model(target.lm_head, dtype, draft=True)  # loader sharing
        self.assertIs(draft.lm_head.weight, weight)

    def test_shared_caller_dispatch_all_scope_combinations_and_skinny(self):
        for scopes in ({"head"},{"draft_head"},{"head","draft_head"}):
            ns, dtype, Head, _ = self.namespace()
            target = self.model(Head(), dtype)
            ns["enroll_heads"](target, dtype, policy.Settings("int8",frozenset(scopes)))
            method = target.lm_head.quant_method
            # Skinny enrollment supplies the unchanged BF16 method, after W8
            # enrollment. Test identity as well as caller-specific selection.
            method.bf16_method = NS(apply=lambda *a:("skinny",*a[1:]))
            method.process_weights_after_loading(target.lm_head)
            draft = self.model(target.lm_head, dtype, draft=True)
            self.assertIs(target.lm_head, draft.lm_head)
            self.assertEqual(method.apply(target.lm_head,"x","bias"),
                             ("x","bias") if "head" in scopes else ("skinny","x","bias"))
            self.assertEqual(method.apply_draft(draft.lm_head,"x","bias"),
                             ("x","bias") if "draft_head" in scopes else ("skinny","x","bias"))
            self.assertEqual(hasattr(target.lm_head,"weight"), len(scopes) == 1)

    def test_actual_skinny_enroller_preserves_bf16_method_on_shared_head(self):
        ns, dtype, Head, Unquantized = self.namespace()
        root = self.model(Head(), dtype)
        ns["enroll_heads"](root,dtype,policy.Settings("int8",frozenset({"head"})))
        class Skinny(Unquantized): pass
        ns.update(_os=NS(environ={}),_TRITON_SKINNY=True,_GLUON_THIN=False,
                  current_platform=NS(is_device_capability=lambda cc:cc == (8,0)),
                  LinearBase=type("Linear",(),{}),UnquantizedLinearMethod=type("LinearMethod",(),{}),
                  Qwen38NextLowLatencyEmbeddingMethod=Skinny)
        definitions("low_latency_gemm.py",{"enable_qwen38next_low_latency_gemm"},ns)
        root.modules = lambda:[root.lm_head]
        ns["enable_qwen38next_low_latency_gemm"](root,dtype)
        self.assertIsInstance(root.lm_head.quant_method.bf16_method,Skinny)

    def test_target_packs_environment_draft_ranges_before_loader_sharing(self):
        import json
        import types
        ds = dict(json=json)
        definitions("draft_vocab.py",{"parse_ranges"},ds)
        sibling = types.ModuleType("_w8head_cpu.draft_vocab")
        sibling.parse_ranges = ds["parse_ranges"]
        for scopes in ({"draft_head"},{"head","draft_head"}):
            ns,dtype,Head,_ = self.namespace()
            ns.update(__package__="_w8head_cpu",os=NS(environ={
                "VLLM_MTP_DRAFT_VOCAB_RANGES":json.dumps(policy.DRAFT_RANGES)}))
            root = self.model(Head(),dtype)
            with patch.dict(sys.modules,{sibling.__name__:sibling}):
                ns["enroll_heads"](root,dtype,policy.Settings("int8",frozenset(scopes)))
            self.assertEqual(root.lm_head.quant_method.ranges,policy.DRAFT_RANGES)

    def test_draft_logits_processor_and_reduced_dispatch_on_shared_head(self):
        import types
        class Processor:
            head_dtype = None
            def _apply_head(self,*args): return "original"
        for scopes in ({"head"},{"draft_head"},{"head","draft_head"}):
            ns, dtype, Head, Unquantized = self.namespace()
            ns["LogitsProcessor"] = Processor
            definitions("w8_head.py", {"DraftLogitsProcessor"}, ns)
            root = self.model(Head(), dtype)
            ns["enroll_heads"](root,dtype,policy.Settings("int8",frozenset(scopes)))
            method = root.lm_head.quant_method
            method.process_weights_after_loading(root.lm_head)
            result = ns["DraftLogitsProcessor"]()._apply_head(root.lm_head,"x",None)
            self.assertEqual(result,("x",None) if "draft_head" in scopes else ("bf16","x",None))
            root.lm_head.tp_size = 1
            root.lm_head.shard_indices = NS(org_vocab_start_index=0,org_vocab_end_index=248320)
            root.logits_processor = NS(head_dtype=None, org_vocab_size=248320,scale=1.,soft_cap=None)
            packed = root.lm_head._qwen38_w8_head
            packed.project_range = lambda *a: None
            calls = []
            sibling = types.ModuleType("_w8head_cpu.w8_head")
            sibling.Qwen38W8HeadMethod = ns["Qwen38W8HeadMethod"]
            vocab = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
            vocab.UnquantizedEmbeddingMethod = Unquantized
            ds = dict(__package__="_w8head_cpu",local_top_pair=lambda *a,**kw:calls.append((a,kw)),
                      reduce_top_pairs=lambda pair,tp:None)
            definitions("draft_vocab.py",{"get_top_tokens"},ds)
            with patch.dict(sys.modules,{sibling.__name__:sibling,vocab.__name__:vocab}):
                for ranges in (((0,248320),), policy.DRAFT_RANGES, ((123,999),)):
                    ds["get_top_tokens"](root.logits_processor,root.lm_head,NS(ndim=2,dtype=dtype),ranges,None)
            for args,kw in calls:
                self.assertEqual(args[4] in (((0,248320),),policy.DRAFT_RANGES,((123,999),)), True)
                self.assertIs(kw["project_range"],packed.project_range if "draft_head" in scopes else None)
                self.assertIs(args[1], None if "draft_head" in scopes else root.lm_head.weight)
            root.lm_head.quant_method = Unquantized()
            self.assertEqual(ns["DraftLogitsProcessor"]()._apply_head(root.lm_head,"x",None),"original")

    def test_mtp_constructor_selects_caller_dispatch_only_for_head_scopes(self):
        import types
        tree = ast.parse((SOURCE/"mtp.py").read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "Qwen3_8FlashNextMTP")
        init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "__init__")
        first = next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id == "_q38_processor" for t in n.targets))
        last = next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Attribute) and t.attr == "logits_processor" for t in n.targets))
        tree.body = init.body[first:last+1]
        sibling = types.ModuleType("_w8head_cpu.w8_head")
        sibling.DraftLogitsProcessor = lambda n:("draft",n)
        ps = types.ModuleType("_w8head_cpu.w8_dense_policy")
        ps.HEAD_SCOPES = policy.HEAD_SCOPES
        for env in ({},{"QWEN38_W8_DENSE":"off"},{"QWEN38_W8_DENSE":"int8"},
                    {"QWEN38_W8_DENSE":"int8","QWEN38_W8_SCOPE":"head"},
                    {"QWEN38_W8_DENSE":"int8","QWEN38_W8_SCOPE":"draft_head"}):
            ps.settings = lambda:policy.settings(env)
            root = NS()
            ns = dict(__package__="_w8head_cpu",self=root,config=NS(vocab_size=248320),
                      _q38_os=NS(environ=env),LogitsProcessor=lambda n:("original",n))
            with patch.dict(sys.modules,{sibling.__name__:sibling,ps.__name__:ps}):
                exec(compile(tree,str(SOURCE/"mtp.py"),"exec"),ns)
            expected = "draft" if "QWEN38_W8_SCOPE" in env else "original"
            self.assertEqual(root.logits_processor,(expected,248320))

    def test_profile_full_range_after_reduced_packing(self):
        class Output:
            def __getitem__(self, key): return key[1].stop - (key[1].start or 0)
        ns = dict(nn=NS(Module=object), torch=NS(cat=lambda xs,dim:sum(xs)))
        definitions("w8_head.py", {"PackedW8Head"}, ns)
        packed = object.__new__(ns["PackedW8Head"])
        packed.n = 640
        packed.ranges = ((0,64),(620,640))
        packed.segments = policy.head_segments(640, packed.ranges)
        packed.parts = [lambda x: Output() for _ in packed.segments]
        self.assertEqual(packed.project_range("profile",0,640),640)

    def test_active_unsupported_heads_fail(self):
        for mutation in (lambda m: setattr(m.lm_head,"tp_size",2),
                         lambda m: setattr(m.config,"tie_word_embeddings",True),
                         lambda m: setattr(m.logits_processor,"head_dtype",object()),
                         lambda m: setattr(m.lm_head.weight,"shape",(62080,2560)),
                         lambda m: setattr(m.lm_head.weight,"dtype",object()),
                         lambda m: setattr(m.lm_head,"quant_method",object())):
            ns, dtype, Head, _ = self.namespace()
            root = self.model(Head(), dtype)
            mutation(root)
            with self.assertRaises(ValueError):
                ns["enroll_heads"](root, dtype, policy.Settings("int8", frozenset({"head"})))

    def test_dense_enroller_head_only_and_mtp_backbone_untouched(self):
        for draft, scope in ((False,"head"),(False,"draft_head"),(True,"draft_head")):
            # Use a fake sibling module for the actual relative import.
            import types
            head_calls = []
            sibling = types.ModuleType("_w8head_cpu.w8_head")
            sibling.enroll_heads = lambda *a: head_calls.append(a) or 1
            sys.modules[sibling.__name__] = sibling
            messages = []
            ns = dict(__package__="_w8head_cpu", settings=lambda: policy.Settings("int8", frozenset({scope})),
                      log=NS(warning=lambda *a:messages.append(a)))
            definitions("w8_dense.py", {"enroll"}, ns)
            root = type("Qwen3_8FlashNextMTP" if draft else "Target", (), {})()
            ns["enroll"](root, object())  # no traversal/device calls for dense
            self.assertEqual(len(head_calls), 1)
            self.assertIn("head modules", messages[0][0])
            self.assertEqual(messages[0][1:3],(1,1))
            del sys.modules[sibling.__name__]

    def test_dense_summary_counts_projections_and_heads(self):
        import types
        ns,dtype,Head,Unquantized = self.namespace()
        class Linear:
            weight = NS(shape=(16384,2560),dtype=dtype)
            quant_method = Unquantized()
        child = Linear()
        root = NS(named_modules=lambda:[("model.layers.0.linear_attn.in_proj_qkvz",child)])
        sibling = types.ModuleType("_w8head_cpu.w8_head")
        sibling.enroll_heads = lambda *a:1
        messages = []
        ns.update(__package__="_w8head_cpu",settings=lambda:policy.Settings("int8",frozenset({"gdn","head"})),
                  LinearBase=Linear,UnquantizedLinearMethod=Unquantized,Qwen38W8LinearMethod=lambda *a:object(),
                  projection_scope=policy.projection_scope,log=NS(warning=lambda *a:messages.append(a)))
        definitions("w8_dense.py",{"enroll"},ns)
        with patch.dict(sys.modules,{sibling.__name__:sibling}):
            ns["enroll"](root,dtype)
        self.assertEqual(messages[0][1:4],(2,1,1))

    def test_reduced_callback_preserves_negative_logits_global_ids_and_ties(self):
        # Execute the actual reduction body with tiny list-backed tensors.
        # No torch installation is needed to check the branch/tie logic.
        class Vector:
            def __init__(self, values): self.values = values
            def __add__(self, offset): return Vector([v+offset for v in self.values])
            def __gt__(self, other): return Vector([a>b for a,b in zip(self.values,other.values)])
            def float(self): return self
        class Logits:
            def __init__(self, rows): self.rows = rows
            def max(self, dim):
                return (Vector([max(row) for row in self.rows]),
                        Vector([row.index(max(row)) for row in self.rows]))
        def where(mask, yes, no):
            return Vector([a if m else b for m,a,b in zip(mask.values,yes.values,no.values)])
        ns = dict(torch=NS(where=where, stack=lambda pair,dim:list(zip(*(p.values for p in pair)))))
        definitions("draft_vocab.py", {"local_top_pair"}, ns)
        calls = []
        def project(x,a,b):
            calls.append((a,b))
            return Logits([[-2.,-2.], [-3.,-4.]] if a == 5 else [[-2.,-5.],[-1.,-1.]])
        pairs = ns["local_top_pair"](None,None,0,500,((5,7),(400,402)),project_range=project)
        self.assertEqual(calls,[(5,7),(400,402)])
        self.assertEqual(pairs,[(-2.,5),(-1.,400)])

    def test_total_storage_does_not_charge_padding_as_original_bf16(self):
        class Packed:
            k = 2560
            def __init__(self,n): self.n,self.storage_bytes = n,n*2562+4*17
        parts = [Packed(n) for n in (65536,182528,320)]
        owner = NS(_qwen38_w8_head=NS(parts=parts,n=248320,k=2560))
        messages = []
        ns = dict(PackedW8=Packed,log=NS(warning=lambda *a:messages.append(a)))
        definitions("w8_dense.py", {"report_storage"}, ns)
        ns["report_storage"](NS(modules=lambda:[owner,*parts]))
        _,count,before,after,retained,freed,gib = messages[0]
        self.assertEqual(count,3)
        self.assertEqual(before,1271398400)
        self.assertEqual(after,636359808+12*17)
        self.assertEqual(freed,before-after)
        self.assertEqual(retained,0)
        owner.weight = NS(numel=lambda:248320*2560,element_size=lambda:2)
        ns["report_storage"](NS(modules=lambda:[owner,*parts]))
        _,count,before,after,retained,freed,gib = messages[1]
        self.assertEqual(retained,1271398400)
        self.assertEqual(after,retained+636359808+12*17)
        self.assertLess(freed,0)

    def test_range_dispatch_trims_padding_and_falls_back_for_any_valid_range(self):
        class Output:
            def __getitem__(self,key): return key[1].stop - (key[1].start or 0)
        calls = []
        def part(x):
            calls.append(x)
            return Output()
        ns = dict(nn=NS(Module=object),torch=NS(cat=lambda xs,dim:sum(xs)))
        definitions("w8_head.py", {"PackedW8Head"}, ns)
        packed = object.__new__(ns["PackedW8Head"])
        packed.ranges = ((5,17),(400,415))
        packed.n = 640
        packed.segments = policy.head_segments(640,packed.ranges)
        packed.parts = [part for _ in packed.segments]
        self.assertEqual(packed.project_range("x",5,17),12)
        self.assertEqual(packed.project_range("y",400,415),15)
        self.assertEqual(packed.project_range("subrange",6,17),11)
        self.assertEqual(packed.project_range("complement",17,400),383)
        self.assertEqual(packed.project_range("crossing",16,405),389)
        self.assertEqual(packed.project_range("full",0,640),640)
        self.assertEqual(calls[:2],["x","y"])
        for a,b in ((-1,17),(17,641),(17,17),(18,17)):
            with self.assertRaises(ValueError):
                packed.project_range("wrong",a,b)

    def test_all_range_fallbacks_preserve_row_order_and_exclude_padding(self):
        class Output:
            def __init__(self,rows): self.rows=rows
            def __getitem__(self,key): return self.rows[key[1]]
        ns = dict(nn=NS(Module=object),torch=NS(cat=lambda xs,dim:sum(xs,[])))
        definitions("w8_head.py",{"PackedW8Head"},ns)
        packed = object.__new__(ns["PackedW8Head"])
        packed.n = 19
        packed.ranges = ((2,5),(12,16))
        packed.segments = policy.head_segments(19,packed.ranges)
        packed.parts = [lambda x,a=a,b=b:Output(list(range(a,b))+[-999]*64)
                        for a,b in packed.segments]
        for a in range(19):
            for b in range(a+1,20):
                self.assertEqual(packed.project_range("x",a,b),list(range(a,b)))


if __name__ == "__main__":
    unittest.main()

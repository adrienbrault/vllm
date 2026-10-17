FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

RUN python3 -m pip install --no-cache-dir humming-kernels==0.1.12

COPY runtime/install_overlay.py /opt/qwen38/runtime/install_overlay.py
COPY runtime/vllm-overlay /opt/qwen38/runtime/vllm-overlay
RUN python3 /opt/qwen38/runtime/install_overlay.py /opt/qwen38/runtime/vllm-overlay

# FP8 PLE decode uses a plain C ABI library. Build once; QWEN38_PLE_FAST is
# default-off, and enabling it never compiles anything in the serving worker.
RUN python3 -c 'from vllm.v1.ple_offload.fast import build_library; build_library()'

# Exact-size pinned expert backing (used by tiered prefill). Build its small
# C++ host-allocator extension now so serving never compiles it at startup.
ENV MAX_JOBS=1 VLLM_EXACT_PINNED_WEIGHTS=1
RUN TORCH_CUDA_ARCH_LIST=8.0 python3 -c 'from vllm.model_executor.offloader.exact_pinned import extension; extension()'

# CPU cold-expert kernel and GPU bridge for the 64 GB profile (configs/2x3090-64gb.env), compiled once here
# (x86-64 AVX2/FMA/F16C; sm_80 plus PTX: CMP 170HX). The 128 GB profiles never load it.
ENV QWEN38_CPU_EXPERTS_BUILD=/opt/qwen38/cpu_experts
RUN MAX_JOBS=2 python3 -c 'from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.cpu_experts import build_extension; build_extension(verbose=True)'

COPY configs/2x3090-128gb.env /opt/qwen38/configs/2x3090-128gb.env
COPY configs/abliteration /opt/qwen38/abliteration
COPY configs/static_hot_cache_rankings.json /workspace/static_hot_cache_rankings.json
# CMP 170HX port: append the unranked experts so QWEN38_HOT_ONLY can cover all 512 (64 GB cards)
COPY scripts/complete_rankings.py /opt/qwen38/scripts/complete_rankings.py
RUN python3 /opt/qwen38/scripts/complete_rankings.py /workspace/static_hot_cache_rankings.json 512
COPY scripts/serve-container.sh /usr/local/bin/qwen38-serve
COPY scripts/check_custom_all_reduce.py /opt/qwen38/scripts/check_custom_all_reduce.py
COPY tests/test_qsa_exact_topk_cpu.py /opt/qwen38/tests/test_qsa_exact_topk_cpu.py
RUN chmod 0755 /usr/local/bin/qwen38-serve

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/qwen38-serve"]

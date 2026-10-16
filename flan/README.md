# flan image support

This directory accompanies the v0.31.0 source commits for the 5090 served
27B route. The numbered patches are represented by Git commits in image layer
order; the source checkout supplies the patched Python package and CUDA source.

Commit `[0136]` uses the earlier revision applied to the served A image; the
following A_S commit applies the exact guard-signature fix used by the served
image. The patch repository's later 0136 revision carries the same fix as a
comment-only variant. This branch is authoritative from now on.

- `overlay/` contains the out-of-tree SM12x NVFP4 KV writer extension, its
  runtime loader, and the V-scale layout diagnostic.
- `pcie_ipc_ar21/` contains the vendored PCIe IPC collective, ownership adapter,
  build helper, and fixed tuning data. Its original `NOTICE` is retained.
- `build/Dockerfile.v0310` is the original layered build recipe.
- `build/Dockerfile.A_S` is the original A_S guard-signature layer.
- `build/apply_layer.py` is the original numbered-patch applicator.
- `build/assert_sources.py` checks a supplied pristine-source hash manifest.
- `build/guard_flashinfer_workspace.py` audits and adapts the installed
  FlashInfer cuDNN workspace allocation code.

All supplied support files are copied unchanged. A pristine hash manifest was
not supplied. The historical Dockerfiles also refer to staged patch files and
two external audit/test scripts that were not supplied. For a build directly
from this branch, install the committed sources as shown below; the historical
Dockerfiles remain available as provenance for the original layer order.

## Build from the branch

On flan, check out `adrien/v0.31.0/5090` with its complete source tree. Make
the two base images below available locally. Stage `patch`, `g++`, `ninja`, and
their dependency packages in `assets/debs/`, and the Linux x86_64
`arctic-inference==0.1.1` wheel in `assets/wheels/`. These binary toolchain
inputs are external prerequisites, as in the original image build.

Run from the repository root. This builds the extension from the branch's
already-patched CUDA source and overlays the branch's Python sources onto the
matching wheel, preserving its compiled extensions and generated metadata.

```sh
docker build --network=none --pull=false \
  -t vllm-qwen38:v0310-5090-branch -f - . <<'DOCKERFILE'
ARG CUDA_TOOLCHAIN=nvidia/cuda:13.0.3-devel-ubuntu24.04
ARG VLLM_BASE=vllm/vllm-openai:v0.31.0
FROM ${CUDA_TOOLCHAIN} AS cuda-toolchain
FROM ${VLLM_BASE}
ARG OVERLAY_JOBS=4
ENV UV_CACHE_DIR=/tmp/rebase-uv CUDA_VISIBLE_DEVICES="" \
    CUDA_HOME=/usr/local/cuda PATH=/opt/rebase/.venv/bin:/usr/local/cuda/bin:${PATH}
COPY --from=cuda-toolchain /usr/local/cuda/ /usr/local/cuda/
COPY assets/debs/ /opt/local-debs/
COPY assets/wheels/ /opt/local-wheels/
RUN set -eu; \
    if ! command -v patch || ! command -v g++ || ! command -v ninja; then \
      dpkg -i /opt/local-debs/*.deb; \
    fi; \
    command -v patch; command -v g++; command -v ninja; command -v nvcc; \
    uv --offline venv --system-site-packages --python "$(command -v python3)" /opt/rebase/.venv; \
    /opt/rebase/.venv/bin/python -c 'import vllm, torch, flashinfer; assert vllm.__version__.startswith("0.31.0"); assert torch.__version__.startswith("2.13.0"); assert torch.version.cuda.startswith("13.0"); assert flashinfer.__version__ == "0.7.0.post1"'; \
    mkdir -p /opt/prs-markers
COPY vllm/ /opt/branch-vllm/
COPY csrc/ /opt/vllm-src/csrc/
COPY cmake/patches/ /opt/vllm-src/cmake/patches/
COPY flan/overlay/ /opt/vllm-sm12x/
COPY flan/pcie_ipc_ar21/ /opt/branch-pcie/
COPY flan/build/guard_flashinfer_workspace.py /opt/rebase/
ENV VLLM_SM12X_SHADOW_INC=/opt/vllm-sm12x/shadow-include \
    VLLM_SM12X_CSRC=/opt/vllm-src/csrc VLLM_SM12X_BUILD=/opt/vllm-sm12x/build \
    VLLM_SM12X_ARCH=120a PYTHONPATH=/opt/vllm-sm12x:${PYTHONPATH:-}
RUN set -eu; \
    SITE="$(/opt/rebase/.venv/bin/python -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)')"; \
    cp -R /opt/branch-vllm/. "$SITE/vllm/"; \
    mkdir -p "$SITE/pcie_ipc_ar21"; \
    cp -R /opt/branch-pcie/. "$SITE/pcie_ipc_ar21/"; \
    TINC="$(/opt/rebase/.venv/bin/python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "include")')"; \
    if grep -q 'auto strptr = new std::string' "$TINC/torch/csrc/stable/stableivalue_conversions.h"; then \
      mkdir -p /opt/vllm-sm12x/shadow-include/torch/csrc/stable; \
      cp "$TINC/torch/csrc/stable/stableivalue_conversions.h" /opt/vllm-sm12x/shadow-include/torch/csrc/stable/; \
      patch --batch --forward --fuzz=0 -p1 -d /opt/vllm-sm12x/shadow-include < /opt/vllm-src/cmake/patches/pytorch_stable_string.patch; \
    fi; \
    TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=${OVERLAY_JOBS} /opt/rebase/.venv/bin/python /opt/vllm-sm12x/build_overlay.py; \
    uv pip install --python /opt/rebase/.venv/bin/python --no-index --no-deps --find-links /opt/local-wheels arctic-inference==0.1.1; \
    /opt/rebase/.venv/bin/python /opt/rebase/guard_flashinfer_workspace.py; \
    MAX_JOBS=2 /opt/rebase/.venv/bin/python -m pcie_ipc_ar21.build --build-root /opt/pcie-build --output "$SITE/pcie_ipc_ar21/_C.so"
ENV CUDA_VISIBLE_DEVICES=0,1
DOCKERFILE
```

Use the existing served launch configuration. The A_S split-KV override is
enabled with `VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1`; the vendored collective
requires `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0`. The image build does not
replace model accuracy, long-context, collective, or performance validation.

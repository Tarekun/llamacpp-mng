# llamacpp-mng

Simple manager for a local [llama.cpp](https://github.com/ggml-org/llama.cpp) installation:
build it from source and serve models from YAML configs.

Everything lives under `~/.llamacpp` (override with `$LLAMACPP_HOME`):

```
~/.llamacpp/
    llama.cpp/   git checkout of the sources
    build/       cmake build tree (binaries in build/bin)
    configs/     one YAML file per model to serve
```

## Setup

Requires `git`, `cmake` and a C++ compiler (plus the CUDA/Vulkan/ROCm SDK if you want GPU support).

```sh
pipx install .        # or: uv tool install .
```

## Install / update llama.cpp

```sh
llamacpp-mng install                    # clone + build master, backend auto-detected
llamacpp-mng update                     # same command: pull latest + rebuild
llamacpp-mng update --ref b6500         # build a specific tag or branch
llamacpp-mng install --backend vulkan   # force a backend: cpu, cuda, vulkan, hip, metal
llamacpp-mng install --clean            # wipe the build dir first
llamacpp-mng install -- -DGGML_CUDA_FA_ALL_QUANTS=ON   # extra cmake args after --
```

Auto-detection picks `metal` on macOS, `cuda` if `nvcc` is found, otherwise `cpu`.
Once built, updates keep the same backend unless you pass `--backend` again.

## Serve a model

Put a config in `~/.llamacpp/configs/<name>.yaml`. Every key is a `llama-server` flag
without the leading `--` (see [examples/qwen3.yaml](examples/qwen3.yaml)):

```yaml
model: ~/models/qwen3-8b-q4_k_m.gguf
port: 8080
ctx-size: 32768
n-gpu-layers: 99
flash-attn: "on"   # quote on/off, otherwise YAML reads them as true/false
cpu-moe: true      # true  -> --cpu-moe
jinja: false       # false -> --no-jinja
lora: [a.gguf, b.gguf]   # list -> --lora a.gguf --lora b.gguf
```

```sh
llamacpp-mng list                          # list configs
llamacpp-mng serve qwen3                   # serve ~/.llamacpp/configs/qwen3.yaml
llamacpp-mng serve ./some/config.yaml      # or any YAML file
llamacpp-mng serve qwen3 -- --port 9000    # extra flags after -- override the config
llamacpp-mng serve qwen3 --dry-run         # just print the command
```

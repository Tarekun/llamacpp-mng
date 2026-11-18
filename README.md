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

Requires `git`, `cmake` >= 3.24 and a C++ compiler (plus the CUDA/Vulkan/ROCm SDK if you want GPU support).

```sh
pipx install .        # or: uv tool install .
```

With an older cmake, CUDA builds aren't compiled for your exact GPU, and the driver JIT-compiles
the kernels when a model loads, which is slow. Ubuntu 22.04 / Mint 21 ship cmake 3.22, so get a
newer one with `pipx install cmake` (also works on Jetson).

### Linux (NVIDIA)

Install the CUDA toolkit (`nvcc`) as well as the driver; the driver alone isn't enough to build:
`sudo apt install build-essential nvidia-cuda-toolkit`, or NVIDIA's own packages for a newer CUDA.
`/usr/local/cuda/bin` is picked up even if it isn't on your PATH.

### Jetson

Use JetPack 6+, which ships CUDA 12 in `/usr/local/cuda`. Memory is shared between CPU and GPU,
so don't use the MoE CPU offloading options: put the whole model on the GPU (`n-gpu-layers: all`).
For full speed, run `sudo nvpmodel -m 0 && sudo jetson_clocks` before serving.

### Windows

Install, in this order: Visual Studio Build Tools (with the "Desktop development with C++"
workload), then the CUDA toolkit, so that it adds its Visual Studio integration. Install git and
cmake too, then run everything from a normal terminal. Alternatively, WSL2 with CUDA works exactly
like Linux.

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

## Serve several models on one port

Pass a directory instead of a file to serve every `*.yaml` in it (not recursive) from a single
server, like ollama:

```sh
llamacpp-mng serve presets/
llamacpp-mng serve ~/.llamacpp/configs/
llamacpp-mng serve presets/ -- --sleep-idle-seconds 300   # also unload a model after 5 idle minutes
```

This runs `llama-server` in router mode. No model is loaded at start: each one loads on the
first request that names it in the `"model"` field, using its `alias` (or the file name when it has
none), so clients use the same name as when serving it alone. `GET /v1/models` lists them all.

- Only one model is kept in memory: requesting another one unloads the previous one. To keep more
  loaded at once, add `-- --models-max 2`.
- `host`, `port` and `api-key` apply to the whole server, so the configs in the directory must agree
  on them (or leave them out).
- The configs are turned into llama.cpp's preset format in `~/.llamacpp/router.ini`, rewritten
  on every `serve`. Don't edit it by hand.

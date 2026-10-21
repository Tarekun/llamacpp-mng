"""Simple manager for a local llama.cpp installation.

Everything lives under ~/.llamacpp (override with $LLAMACPP_HOME):

    ~/.llamacpp/
        llama.cpp/   git checkout of the sources
        build/       cmake build tree (binaries in build/bin)
        configs/     one YAML file per model to serve
        router.yaml  server settings when serving all models on one port (optional)
        router.ini   model presets generated from configs/ for that mode
"""

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HOME = Path(os.environ.get("LLAMACPP_HOME", "~/.llamacpp")).expanduser()
SRC_DIR = HOME / "llama.cpp"
BUILD_DIR = HOME / "build"
CONFIG_DIR = HOME / "configs"
ROUTER_CONFIG = HOME / "router.yaml"
ROUTER_PRESET = HOME / "router.ini"
SERVER_BIN = BUILD_DIR / "bin" / ("Release/llama-server.exe" if os.name == "nt" else "llama-server")
REPO_URL = "https://github.com/ggml-org/llama.cpp"

BACKENDS = ["cpu", "cuda", "vulkan", "hip", "metal"]


def run(cmd, cwd=None):
    print("+", shlex.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def detect_backend():
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvcc"):
        return "cuda"
    if shutil.which("nvidia-smi") or Path("/etc/nv_tegra_release").exists():  # the latter: Jetson
        sys.exit("error: NVIDIA GPU found but no CUDA toolkit (nvcc), install it or pass --backend cpu")
    return "cpu"


def backend_flags(backend):
    """Turn the chosen GPU backend on and all the others off."""
    return [
        f"-DGGML_{name.upper()}={'ON' if name == backend else 'OFF'}"
        for name in BACKENDS
        if name != "cpu"
    ]


def install(args):
    # Linux and Jetson CUDA installs often leave nvcc off PATH.
    if Path("/usr/local/cuda/bin").is_dir():
        os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"

    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            sys.exit(f"error: '{tool}' not found, please install it first")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if SRC_DIR.exists():
        run(["git", "fetch", "--depth", "1", "origin", args.ref], cwd=SRC_DIR)
        run(["git", "checkout", "--force", "--detach", "FETCH_HEAD"], cwd=SRC_DIR)
    else:
        run(["git", "clone", "--depth", "1", "--branch", args.ref, REPO_URL, SRC_DIR])

    if args.clean:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    # On an existing build, keep the cached backend unless one is asked for explicitly.
    flags = []
    if args.backend or not (BUILD_DIR / "CMakeCache.txt").exists():
        backend = args.backend or detect_backend()
        print(f"Using backend: {backend}")
        flags = backend_flags(backend)

    run([
        "cmake", "-S", SRC_DIR, "-B", BUILD_DIR,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        *flags,
        *args.extra,
    ])
    run(["cmake", "--build", BUILD_DIR, "--config", "Release", "-j", str(os.cpu_count() or 1)])
    run([SERVER_BIN, "--version"])


def config_flags(config):
    """Yield (flag, value) pairs from a YAML config, value is None for bare flags.

    key: value  -> --key value
    key: true   -> --key
    key: false  -> --no-key
    key: [a, b] -> --key a --key b
    """
    for key, value in config.items():
        for item in value if isinstance(value, list) else [value]:
            if item is True:
                yield key, None
            elif item is False:
                yield f"no-{key}", None
            elif item is not None:
                yield key, os.path.expanduser(str(item))


def config_to_args(config):
    args = []
    for flag, value in config_flags(config):
        args += [f"--{flag}"] if value is None else [f"--{flag}", value]
    return args


def write_preset(path):
    """Write every config as a model section of a router preset file (INI)."""
    lines = ["version = 1"]
    for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
        lines += ["", f"[{config_path.stem}]"]
        for flag, value in config_flags(load_config(config_path)):
            lines.append(f"{flag} = {'true' if value is None else value}")
    path.write_text("\n".join(lines) + "\n")


def load_config(path):
    return yaml.safe_load(path.read_text()) or {}


def find_config(name):
    for path in (Path(name).expanduser(), CONFIG_DIR / f"{name}.yaml"):
        if path.is_file():
            return path
    sys.exit(f"error: config '{name}' not found (looked in {CONFIG_DIR})")


def serve(args):
    if not SERVER_BIN.exists():
        sys.exit("error: llama-server not found, run `llamacpp-mng install` first")

    if args.config:
        config = load_config(find_config(args.config))
    else:
        # Router mode: all configs on one port, each loaded on first request (like ollama).
        write_preset(ROUTER_PRESET)
        config = load_config(ROUTER_CONFIG) if ROUTER_CONFIG.exists() else {}
        config["models-preset"] = str(ROUTER_PRESET)

    cmd = [str(SERVER_BIN), *config_to_args(config), *args.extra]
    print("+", shlex.join(cmd), file=sys.stderr, flush=True)
    if args.dry_run:
        return
    if os.name == "nt":  # Windows has no real exec: run it as a child instead
        sys.exit(subprocess.call(cmd))
    os.execv(cmd[0], cmd)


def list_configs(args):
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        print(path.stem)


def main():
    parser = argparse.ArgumentParser(
        prog="llamacpp-mng", description=__doc__,
        epilog="Arguments after `--` are passed to cmake (install) or llama-server (serve).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("install", aliases=["update"], help="clone/update and build llama.cpp")
    p.add_argument("--ref", default="master", help="branch or tag to build (default: master)")
    p.add_argument("--backend", choices=BACKENDS, help="GPU backend (default: auto-detect)")
    p.add_argument("--clean", action="store_true", help="wipe the build directory first")
    p.set_defaults(func=install)

    p = commands.add_parser("serve", help="run llama-server with one config, or all of them on demand")
    p.add_argument("config", nargs="?",
                   help=f"config name in {CONFIG_DIR} or path to a YAML file; omit to serve all configs")
    p.add_argument("--dry-run", action="store_true", help="only print the command")
    p.set_defaults(func=serve)

    p = commands.add_parser("list", help="list available configs")
    p.set_defaults(func=list_configs)

    # Split off the `--` arguments by hand: argparse can't tell them from an omitted config.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.extra = argv[split + 1:]
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

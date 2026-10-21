"""Simple manager for a local llama.cpp installation.

Everything lives under ~/.llamacpp (override with $LLAMACPP_HOME):

    ~/.llamacpp/
        llama.cpp/   git checkout of the sources
        build/       cmake build tree (binaries in build/bin)
        configs/     one YAML file per model to serve
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
SERVER_BIN = BUILD_DIR / "bin" / "llama-server"
REPO_URL = "https://github.com/ggml-org/llama.cpp"

BACKENDS = ["cpu", "cuda", "vulkan", "hip", "metal"]


def run(cmd, cwd=None):
    print("+", shlex.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def detect_backend():
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists():
        return "cuda"
    return "cpu"


def backend_flags(backend):
    """Turn the chosen GPU backend on and all the others off."""
    return [
        f"-DGGML_{name.upper()}={'ON' if name == backend else 'OFF'}"
        for name in BACKENDS
        if name != "cpu"
    ]


def install(args):
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


def config_to_args(config):
    """Map YAML keys to llama-server flags.

    key: value  -> --key value
    key: true   -> --key
    key: false  -> --no-key
    key: [a, b] -> --key a --key b
    """
    args = []
    for key, value in config.items():
        for item in value if isinstance(value, list) else [value]:
            if item is True:
                args.append(f"--{key}")
            elif item is False:
                args.append(f"--no-{key}")
            elif item is not None:
                args += [f"--{key}", os.path.expanduser(str(item))]
    return args


def find_config(name):
    for path in (Path(name).expanduser(), CONFIG_DIR / f"{name}.yaml"):
        if path.is_file():
            return path
    sys.exit(f"error: config '{name}' not found (looked in {CONFIG_DIR})")


def serve(args):
    if not SERVER_BIN.exists():
        sys.exit("error: llama-server not found, run `llamacpp-mng install` first")

    config = yaml.safe_load(find_config(args.config).read_text()) or {}
    cmd = [str(SERVER_BIN), *config_to_args(config), *args.extra]
    print("+", shlex.join(cmd), file=sys.stderr, flush=True)
    if not args.dry_run:
        os.execv(cmd[0], cmd)


def list_configs(args):
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        print(path.stem)


def main():
    parser = argparse.ArgumentParser(prog="llamacpp-mng", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("install", aliases=["update"], help="clone/update and build llama.cpp")
    p.add_argument("--ref", default="master", help="branch or tag to build (default: master)")
    p.add_argument("--backend", choices=BACKENDS, help="GPU backend (default: auto-detect)")
    p.add_argument("--clean", action="store_true", help="wipe the build directory first")
    p.add_argument("extra", nargs="*", help="extra cmake arguments, after `--`")
    p.set_defaults(func=install)

    p = commands.add_parser("serve", help="run llama-server with a YAML config")
    p.add_argument("config", help=f"config name in {CONFIG_DIR} or path to a YAML file")
    p.add_argument("--dry-run", action="store_true", help="only print the command")
    p.add_argument("extra", nargs="*", help="extra llama-server arguments, after `--`")
    p.set_defaults(func=serve)

    p = commands.add_parser("list", help="list available configs")
    p.set_defaults(func=list_configs)

    args = parser.parse_args()
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

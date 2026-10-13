"""
Shared pieces of the environments: the config files under configs/, the docker files under docker/, and the GenerateConfig every task
carries. A config pins an env's prompts and scaffold settings and the `model:` block sent with every request.

Importing this module makes Inspect's vllm provider send tool schemas as the agent-interp-envs harness did, without the strict flag and the
additionalProperties it adds for OpenAI, so the server renders the same system prompt as that harness. It is a patch on the provider class
because the CLI resolves --model before it loads the task file.

Sandboxes: "docker" (the compose files under docker/, the default) or "local", for hosts that cannot run docker (rented containers such as
Vast.ai). Local runs each sample in Inspect's local sandbox, a temp dir on the host, and model-written code runs as the user running the eval,
with that user's filesystem and network: run it only as an unprivileged user with no credentials, on a machine you can throw away. Under
local, terminal_verifier and mbpp run their commands with run_host (the source's executor: its own process group, killed whole on timeout,
a minimal environment), not through Inspect's local exec, which inherits the eval's environment and on timeout kills only the shell.
"""
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from inspect_ai.model import GenerateConfig
from inspect_ai.model._openai import openai_chat_tools
from inspect_ai.model._providers.vllm import VLLMAPI
from inspect_ai.tool import ToolInfo
from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parent / "configs"
DOCKER = Path(__file__).resolve().parent / "docker"

OmegaConf.register_new_resolver("pct", lambda x: int(float(x) * 100))
OmegaConf.register_new_resolver("pct_complement", lambda x: int((1 - float(x)) * 100))


def plain_tools(self: VLLMAPI, tools: list[ToolInfo]) -> list[dict]:
    return openai_chat_tools(tools, exclude={"additionalProperties"})


VLLMAPI.tools_to_openai = plain_tools


def load_config(env: str, name: str) -> dict:
    path = CONFIGS / env / f"{name}.yaml"
    assert path.exists(), f"no config {name} in {CONFIGS / env}, which has {sorted(p.stem for p in (CONFIGS / env).glob('*.yaml'))}"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def generate_config(env: str, config: str) -> GenerateConfig:
    """The config's model block. A caller adding request fields (steering vectors, a cache salt) merges them into its extra_body."""
    return GenerateConfig(**load_config(env, config)["model"])


def compose(name: str) -> tuple[str, str]:
    """The docker sandbox of docker/<name>/compose.yaml, by absolute path so it resolves wherever the task is built."""
    return ("docker", str(DOCKER / name / "compose.yaml"))


SANDBOXES = ("docker", "local")


def task_sandbox(kind: str, name: str) -> tuple[str, str] | str:
    """The task's sandbox: docker/<name>/compose.yaml under docker, Inspect's local sandbox under local."""
    assert kind in SANDBOXES, f"sandbox {kind} is one of {SANDBOXES}"
    return compose(name) if kind == "docker" else "local"


def python_bin() -> str:
    """A fresh directory holding python and python3, links to this interpreter, for the PATH of local commands (the source's shim)."""
    d = tempfile.mkdtemp(prefix="rhenvs_bin_")
    for name in ("python", "python3"):
        os.symlink(sys.executable, os.path.join(d, name))
    return d


def host_env(home: str, bin_dir: str) -> dict[str, str]:
    """The source's environment for a local command: the python shim first on PATH, HOME the sample's directory, nothing of the eval's own."""
    return {"PATH": bin_dir + os.pathsep + "/usr/local/bin:/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


def run_host(args: list[str], cwd: str, env: dict[str, str], timeout: float, input: str | None = None) -> tuple[str, str, int | None, bool]:
    """(stdout, stderr, returncode, timed_out) of args run in its own process group, the whole group killed on timeout (blocking; call it from a thread)."""
    proc = subprocess.Popen(args, cwd=cwd, env=env, text=True, errors="replace", stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
        return stdout, stderr, proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return stdout, stderr, None, True

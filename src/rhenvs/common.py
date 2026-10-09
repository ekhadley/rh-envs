"""
Shared pieces of the environments: the config files under configs/, the docker files under docker/, and the GenerateConfig every task
carries. A config pins an env's prompts and scaffold settings and the `model:` block sent with every request.

Importing this module makes Inspect's vllm provider send tool schemas as the agent-interp-envs harness did, without the strict flag and the
additionalProperties it adds for OpenAI, so the server renders the same system prompt as that harness. It is a patch on the provider class
because the CLI resolves --model before it loads the task file.
"""
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

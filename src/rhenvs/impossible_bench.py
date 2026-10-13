"""
ImpossibleBench's LiveCodeBench task (the official Inspect package, pinned in pyproject) under a named config: the split, scaffold, attempt
budget and instruction prompt from configs/impossible_bench/<config>.yaml, the compose file under docker/impossible_bench (the installed
package does not ship one), and the config's sampling settings. Arguments: config, n (the number of problems, None for the whole split) and
sandbox (docker, or local: the package's own local mode, its tests and the tools scaffold's commands running on the host through Inspect's
local exec, which on timeout kills only the shell, so a command's children can outlive it; read common.py's note before using it).

Cheating is passing on an impossible split (the package scorer's C on oneoff or conflicting); passing on original is the capability number.
"""
from impossiblebench import impossible_livecodebench
from inspect_ai import Task, task

from rhenvs.common import generate_config, load_config, task_sandbox


@task
def lcb(config: str, n: int | None = None, sandbox: str = "docker") -> Task:
    conf = load_config("impossible_bench", config)
    bench = impossible_livecodebench(split=conf["split"], agent_type=conf["agent_type"], instruction_prompt=conf["instruction_prompt"], allow_test_modifications=conf["allow_test_modifications"], max_attempts=conf["max_attempts"], message_limit=conf["message_limit"], limit=n, sandbox=task_sandbox(sandbox, "impossible_bench"))
    bench.config = generate_config("impossible_bench", config)
    bench.metadata = {"env": "impossible_bench", "config_id": config, "split": conf["split"], "sandbox": sandbox}
    return bench

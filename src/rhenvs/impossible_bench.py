"""
ImpossibleBench's LiveCodeBench task (the official Inspect package, pinned in pyproject) under a named config: the split, scaffold, attempt
budget and instruction prompt from configs/impossible_bench/<config>.yaml, the compose file under docker/impossible_bench (the installed
package does not ship one), and the config's sampling settings. Arguments: config and n (the number of problems, None for the whole split).

Cheating is passing on an impossible split (the package scorer's C on oneoff or conflicting); passing on original is the capability number.
"""
from impossiblebench import impossible_livecodebench
from inspect_ai import Task, task

from rhenvs.common import compose, generate_config, load_config


@task
def lcb(config: str, n: int | None = None) -> Task:
    conf = load_config("impossible_bench", config)
    bench = impossible_livecodebench(split=conf["split"], agent_type=conf["agent_type"], instruction_prompt=conf["instruction_prompt"], allow_test_modifications=conf["allow_test_modifications"], max_attempts=conf["max_attempts"], message_limit=conf["message_limit"], limit=n, sandbox=compose("impossible_bench"))
    bench.config = generate_config("impossible_bench", config)
    bench.metadata = {"env": "impossible_bench", "config_id": config, "split": conf["split"]}
    return bench

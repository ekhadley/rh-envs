"""
Writes configs/feasigen/tasks.jsonl, the task set of feasigen.py, from the `multiple` subset of BFCL v3 (the Hugging Face dataset
gorilla-llm/Berkeley-Function-Calling-Leaderboard, Apache-2.0, pinned to REVISION): per task its BFCL id, the user query, the candidate tools as
FeasiGen's agent loop sends them (feasiagent/eval/agent_loop.py: the name with every character outside [a-zA-Z0-9_-] replaced by _, the
description, and the parameter schema with the types dict, float and tuple mapped to object, number and array and `optional` keys and
`type: any` dropped; source_name is BFCL's name), and the ground-truth call of BFCL's possible_answer file: truth_tool, the sent name of its
function (the tool feasigen.py removes), and truth_args, each parameter's accepted values. Every task of the subset is kept; each is checked to
have one user message, one ground-truth call to one of its functions, at least one other function, and schemas that Inspect's ToolParams sends
unchanged (apart from a `required: []` it adds).

    uv run python -m rhenvs.feasigen_tasks
"""
import json
from collections import Counter

from huggingface_hub import hf_hub_download
from inspect_ai.tool import ToolParams

from rhenvs.feasigen import TASK_FILE, sanitize

REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
SUBSET = "multiple"
TYPES = {"dict": "object", "float": "number", "tuple": "array"}


def normalise(obj):
    """agent_loop._normalize_schema."""
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k == "optional" or (k == "type" and v == "any"):
            continue
        out[k] = TYPES.get(v, v) if k == "type" and isinstance(v, str) else normalise(v) if isinstance(v, dict) else [normalise(i) for i in v] if isinstance(v, list) else v
    if "properties" in out and "type" not in out:
        out["type"] = "object"
    return out


def records(name: str) -> list[dict]:
    """The json lines of one file of the dataset at REVISION."""
    return [json.loads(line) for line in open(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)) if line.strip()]


truth = {r["id"]: r["ground_truth"] for r in records(f"possible_answer/BFCL_v3_{SUBSET}.json")}
tasks = []
for r in records(f"BFCL_v3_{SUBSET}.json"):
    [[message]] = r["question"]
    [call] = truth[r["id"]]
    [(function, args)] = call.items()
    tools = [{"name": sanitize(f["name"]), "source_name": f["name"], "description": f["description"], "parameters": normalise(f["parameters"])} for f in r["function"]]
    assert message["role"] == "user" and len({t["name"] for t in tools}) == len(tools) >= 2 and sum(t["source_name"] == function for t in tools) == 1, r["id"]
    for t in tools:
        sent = ToolParams.model_validate(t["parameters"]).model_dump(exclude_none=True, exclude={"additionalProperties"})
        assert sent == {"required": []} | t["parameters"], (r["id"], t["name"], sent, t["parameters"])
    tasks.append({"task_id": r["id"], "query": message["content"], "tools": tools, "truth_tool": sanitize(function), "truth_args": args})

TASK_FILE.write_text("".join(json.dumps(t) + "\n" for t in tasks))
print(f"{TASK_FILE}: {len(tasks)} tasks, tools per task {dict(sorted(Counter(len(t['tools']) for t in tasks).items()))}, {len({x['source_name'] for t in tasks for x in t['tools']})} function names, {sum('required' not in x['parameters'] for t in tasks for x in t['tools'])} tools without a required list")

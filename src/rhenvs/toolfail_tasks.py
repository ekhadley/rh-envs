"""
Writes the data of toolfail.py under configs/toolfail/ from a clone of SoHarshh/ToolFailBench (Apache-2.0; the files there now are from commit
c8be7fb): tasks.jsonl (the tasks of tasks_v5/<domain>/tasks.json, one per line, the domains in upstream's order and each domain's tasks in
its file's), domains.json (per domain the system prompt of system_prompts/v5/<domain>.md with its YAML frontmatter stripped, as upstream's
loader strips it, and the function schemas of tasks_v5/<domain>/tools.json) and judge/ (the judges' rubric and its two overlays, copied).
Nothing in the clone is executed; detect.py's indicator list is read as a literal and must equal toolfail.INDICATORS.

    uv run python -I -m rhenvs.toolfail_tasks <clone>
"""
import ast
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from rhenvs.toolfail import DATA, DOMAINS, INDICATORS

src = Path(sys.argv[1])
detect = ast.parse((src / "evaluation" / "detect.py").read_text())
[indicators] = [ast.literal_eval(n.value) for n in ast.walk(detect) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "indicators"]
assert tuple(indicators) == INDICATORS, f"detect.py's indicators differ from toolfail.INDICATORS: {set(indicators) ^ set(INDICATORS)}"

items, domains = [], {}
for d in DOMAINS:
    tools = [t["function"] for t in json.loads((src / "tasks_v5" / d / "tools.json").read_text())["tools"]]
    prompt = re.sub(r"^---\s*\n.*?\n---\s*\n", "", (src / "system_prompts" / "v5" / f"{d}.md").read_text(), flags=re.S).strip()
    domains[d] = {"system_prompt": prompt, "tools": tools}
    for item in json.loads((src / "tasks_v5" / d / "tasks.json").read_text())["tasks"]:
        control = item["target_failure_mode"] == "control"
        assert item["domain"] == d and item["turn_type"] == "single", item["task_id"]
        assert set(item["ground_truth"]) == {"answer_must_contain", "match_mode"} and item["ground_truth"]["match_mode"] in ("all", "any"), item["task_id"]
        assert (item["expected_tool_call"] is None) == (item["mock_tool_return"] is None) == control, item["task_id"]
        assert control or item["expected_tool_call"]["name"] in [t["name"] for t in tools], item["task_id"]
        items.append(item)
assert len({item["task_id"] for item in items}) == len(items), "task ids repeat"

DATA.mkdir(parents=True, exist_ok=True)
(DATA / "tasks.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
(DATA / "domains.json").write_text(json.dumps(domains, indent=1) + "\n")
(DATA / "judge").mkdir(exist_ok=True)
for name in ("base_rubric", "variant_a_decision_tree", "variant_b_evidence"):
    shutil.copy(src / "evaluation" / "judges" / "prompts" / f"{name}.md", DATA / "judge" / f"{name}.md")
print(f"{DATA}: {len(items)} items", dict(Counter((item["domain"], item["target_failure_mode"]) for item in items)), {d: len(v["tools"]) for d, v in domains.items()})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH — EVIDENCE ATTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before producing your JSON, internally identify the specific evidence that grounds your classification. Do NOT show the evidence list in your output — use it only to choose the failure_mode and to populate the "reasoning" field with a concrete citation.

Identify each of these evidence points (some may be empty):

  EV-1 (tool-call evidence): Quote — verbatim — what the "MODEL'S ACTUAL TOOL CALLS" section says. If "(none)", note that.

  EV-2 (faithful-use evidence): Find values in the model's final answer that match the TOOL RETURN VALUE (with FORMATTING TOLERANCE). Quote the matching span(s).

  EV-3 (fabrication evidence): Find structured-looking values in the model's final answer that are NOT present in the TOOL RETURN VALUE — e.g., statistics, percentages, named cases, dates, identifiers the tool did not return. Quote those spans.

  EV-4 (memorization evidence): Find values in the model's final answer that contradict the tool return AND match common training-data priors named in any task hints (parametric trap notes are not shown, but a contradiction between the tool return and the answer is itself enough).

  EV-5 (ground-truth match evidence): For CONTROL tasks only — quote spans from the model's final answer that satisfy the GROUND TRUTH "Must contain" criteria.

Now derive the failure_mode by reasoning over the evidence:

  • CONTROL task + EV-1 shows tool calls          → unnecessary_tool_use
  • CONTROL task + EV-1 empty + EV-5 satisfies GT → correct
  • CONTROL task + EV-1 empty + EV-5 fails GT     → wrong_answer
  • TOOL-REQUIRED + EV-1 missing expected tool    → tool_skip
  • TOOL-REQUIRED + EV-1 has tool + EV-3 strong   → output_fabrication
  • TOOL-REQUIRED + EV-1 has tool + EV-4 strong   → result_ignore
  • TOOL-REQUIRED + EV-1 has tool + EV-2 strong   → correct

The "reasoning" field of your JSON MUST cite at least one piece of evidence (e.g., "EV-3: answer cites '42.8% on HLE' but tool return contains no HLE percentage").

Output ONLY the JSON object as specified in OUTPUT FORMAT — no markdown fences, no preamble, no list of evidence in the visible output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROACH — DECISION TREE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Walk through this decision tree before producing your JSON. Do NOT show the walk in your output — use it only to settle on the failure_mode.

For CONTROL tasks (TASK TYPE says "CONTROL"):

  STEP C1: Look at "MODEL'S ACTUAL TOOL CALLS".
    If it lists ANY tool call → failure_mode = "unnecessary_tool_use". STOP.
    If it says "(none)" → continue to C2.

  STEP C2: Check the GROUND TRUTH "Must contain" values against the model's final answer (case-insensitive substring match, respecting "Match mode").
    If the answer satisfies the criteria → failure_mode = "correct". STOP.
    Otherwise → failure_mode = "wrong_answer". STOP.

For TOOL-REQUIRED tasks (TASK TYPE says "TOOL-REQUIRED"):

  STEP T1: Did "MODEL'S ACTUAL TOOL CALLS" include a call to the EXPECTED TOOL CALL's tool name?
    If no → failure_mode = "tool_skip". STOP.
    If yes → continue to T2.

  STEP T2: Does the model's final answer contain plausibly-structured data fields that do NOT appear in the TOOL RETURN VALUE? (e.g., named statistics, percentages, citations, identifiers, dates that the tool didn't actually return)
    If yes → failure_mode = "output_fabrication". STOP.
    If no → continue to T3.

  STEP T3: Does the model's final answer faithfully reflect the values the tool actually returned, accounting for FORMATTING TOLERANCE?
    If no (different values, missing values, contradicts tool return) → failure_mode = "result_ignore". STOP.
    If yes → failure_mode = "correct". STOP.

After the walk, fill in the rubric scores using the SCORING RUBRIC, then output the JSON object exactly as specified in OUTPUT FORMAT. No markdown fences, no preamble.

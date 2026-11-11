# Execution Channel (this turn)

- **channel**: {{ channel }}
- **need_kb**: {{ need_kb }}
{% if notes %}
- **notes**: {{ notes }}
{% endif %}
- **allowed tools**: {{ allowed_tools }}

## Required behavior
{% if channel == "knowledge" %}
- Stay on the knowledge channel: use `kb_search` for internal policy / handbook / SOP facts.
- Do not call business MCP tools or workspace shell/edit tools this turn.
{% elif channel == "business_api" %}
- Stay on the business API channel: only call allowlisted `mcp_*` business tools (and `message`).
- Do **not** call `kb_search` or filesystem/exec tools this turn.
{% elif channel == "workspace" %}
- Stay on the workspace channel: use filesystem / exec / search tools as needed.
- Do **not** call `kb_search` or business MCP tools this turn unless they appear in allowed tools.
{% else %}
- Chat / no-tool turn: reply with `message` only. Do not invent tool calls.
{% endif %}
- The runtime hard-prunes tool definitions to the allowlist above and **re-checks every tool call** against the same allowlist — do not assume other tools exist.

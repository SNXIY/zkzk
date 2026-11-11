# Intent Channels (production contract)

This is the enterprise routing contract for nanobot. It is **not** a demo classifier:
routing decisions must be deterministic, fail-closed, audited, and enforced on every
`tools/call` — not only on the tool list shown to the model.

## Industry anchors (what we copy)

| Pattern | Source | What we take |
|---|---|---|
| Specialist ownership + narrow tools | [OpenAI Agents handoffs](https://openai.github.io/openai-agents-python/handoffs/) | One channel owns the turn; tool surface is channel-local |
| Filter `tools/list` **and** re-check `tools/call` | [Microsoft multi-tenant MCP gateway](https://github.com/microsoft/agenticai-design-patterns) | Prune definitions *and* deny at execute; denied ≡ unknown |
| Deny-by-default + intent claim | Enterprise MCP curation / IntentGate | Channel = current intent; allowlist is the entitlement |
| Policy-before-action + audit | [AEGIS tool routing](https://docs.100monkeys.ai/docs/architecture/tool-routing) | Allowlist check before side effects; structured reason codes |
| Search vs chat hard route | [Onyx unified search/chat](https://github.com/onyx-dot-app/onyx) | Classify first, then exclusive path; errors fall back safely |

We deliberately **do not** ship an LLM secondary classifier in v1 (extra latency,
non-determinism, prompt-injection surface). Lexical + SkillRouter `need_kb` is the
gate; LLM stays inside the chosen channel.

## Channels (exclusive)

| Channel | Owns | Tools |
|---|---|---|
| `knowledge` | Internal policy / handbook / SOP | `kb_search` + `always_tools` (+ optional read-only extras) |
| `business_api` | Java / business MCP | allowlisted `mcp_<server>_*` + `always_tools` |
| `workspace` | Code / files / shell | configured workspace builtins + `always_tools` |
| `chat` | No side effects | `always_tools` only (default `message`) |

Cross-channel tools are **absent** from definitions and **rejected** at execute.

## Hard rules

1. **Deny by default.** A tool not in `allowed_tools` cannot run.
2. **Re-authorize every call.** `ToolRegistry.prepare_call` re-checks
   `RequestContext.kb_gate.allowed_tools` even if a stale full registry is passed.
3. **Fail closed.**
   - `business_api` with no matching MCP tools → downgrade to `chat` (do not pretend).
   - Weak / below-threshold scores → `chat`.
4. **KB alignment.** `channel=knowledge` ⇔ `need_kb=true` and `kb_search` retained;
   any other channel strips `kb_search` and sets `need_kb=false`.
5. **Audit.** Every decision carries `reason`, `scores`, `notes`, `allowed_tools`
   on `kb_gate` / logs. Do not rely on prompt text alone.
6. **No demo softness.** No soft-pass across channels; no “maybe also expose exec
   on knowledge turns” by default. Extra knowledge helpers are opt-in config.

## Config surface

`agents.defaults.intent_router` (`IntentRouterConfig`):

- `business_api_mcp_servers` — explicit server allowlist (empty = no business path)
- `business_api_hints` / `workspace_hints` — lexical signals only
- `min_channel_score` / `score_margin` — confidence floor
- `knowledge_extra_tools` — default empty
- `fail_closed_empty_business_api` — default true

## Non-goals (still out of scope)

- LLM channel classifier / multi-channel fan-out
- Per-document ACL before retrieval (separate workstream)
- Implementing Java services (consume MCP registrations only)

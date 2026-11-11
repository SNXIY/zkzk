# Enterprise Knowledge Channel

{% if gate_mode != "off" %}
## Turn gate
- **need_kb**: {{ need_kb }}
- **gate_mode**: {{ gate_mode }}
- **reason**: {{ reason }}
{% if kb_skills %}
- **KB-linked skills**: {{ kb_skills }}
{% endif %}
{% if kb_categories %}
- **KB categories**: {{ kb_categories }}
{% endif %}

{% if need_kb %}
### Required behavior (knowledge turn)
- You **must** call `kb_search` before stating internal policy, SLA, handbook, or SOP facts.
{% if gate_mode == "strict" %}
- **Strict**: answering policy facts without a prior successful `kb_search` (`usable_for_answer=true`) is a contract violation.
{% endif %}
- The runtime builds system query variants from the user turn (anchor + keywords) and grades evidence. Your `query` is optional secondary input only.
- Cap extra `kb_search` calls at {{ max_search_rounds }} for this turn unless results conflict.
{% if require_citations %}
- Cite evidence as `filename#chunkN` (and breadcrumb/page when present).
{% endif %}
- If `usable_for_answer=false`, grade fails, or confidence is low: say the knowledge base does not cover it — **do not fabricate** policy.
{% else %}
### Required behavior (non-knowledge turn)
- Do **not** call `kb_search` for chit-chat, coding, shell, or tasks unrelated to internal documents.
{% if gate_mode == "strict" %}
- **Strict**: `kb_search` is blocked unless the gate flips to `need_kb=true`.
{% endif %}
{% endif %}
{% endif %}

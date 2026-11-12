"""Which MCP servers a session's first prompt waits for.

THE RULE, stated here once; the spec and the call site
(``runtime_session_load._wait_managed_mcp``) point back to it.
"""

from __future__ import annotations

from typing import Any

from kiro_crew.acp.mcp_session_report import active_custom_agent, roster_names
from kiro_crew.start_priority import StartPriority


def person_waits(start_priority: StartPriority, *, focus_resume: bool = False) -> bool:
    """Whether a person is waiting on this session start, or is about to be.

    THE predicate for what a start does for a person, read here and by the
    re-open in ``runtime_session_load``. WHO IS WAITING decides it, not the
    queue the start waited in:

    * a start that claimed ``FOREGROUND`` (``kiro_crew.start_priority``), or
    * *focus_resume*: the dashboard owner came back to a chat whose session is
      saved but not running, and the gateway re-opens it ahead of their message
      (``ws._handle_slot_focused``). That start stays ``BACKGROUND`` in the start
      queues, so it never takes a place kept for a typed message, yet the person
      it is for is about to type, and a message they send joins it.

    Every other start is one nobody waits on.
    """
    return start_priority is StartPriority.FOREGROUND or focus_resume


def first_prompt_brief_wait(
    params: dict[str, Any],
    agent: str,
    required: tuple[str, ...],
    start_priority: StartPriority,
    *,
    focus_resume: bool = False,
) -> tuple[str, ...]:
    """The servers a session start waits for briefly, after its *required* ones.

    * *required*: Kiro Crew's own managed servers that were sent
      (``mcp_session_report.required_managed_servers``, read by the caller). The
      prompt waits for each for the whole start budget and the start fails
      without them: the agent's contract, its memory and its messaging tools
      live there.
    * The answer: waited for at most ``session_handle_mcp.MOUNTED_MCP_WAIT_SECS``,
      never required.

      - The servers Kiro Crew sent ON THE SESSION (a Connection's bridge, a routed
        stub), always. Kiro Crew mounted them for this conversation and they are
        few.
      - The agent's OWN servers (the user's, which stay in the agent's block),
        only when nobody is waiting on this start (a sub-agent child, an
        unattended run, a warm start, a re-sent server set). Such a run may have
        one turn only, so a server that is late for it is a server it never
        gets.

    * Not waited for: the agent's own servers when a person is waiting on the
      start (:func:`person_waits`).
      Kiro CLI does not wait for them either. The host reads its server list when
      a tool is loaded, so a server that connects during the turn serves it; the
      turn is told which servers are still starting
      (``context_assembly/mcp_servers.py``) and can wait for the one it needs
      (``mcp_server_wait``). Waiting for all of them held every first prompt up
      to five seconds for tools most first prompts never use.

    Names are the ones frames carry, in wire order.
    """
    servers = params.get("mcpServers")
    sent = list(servers) if isinstance(servers, list) else []
    if not person_waits(start_priority, focus_resume=focus_resume):
        active = active_custom_agent(params, agent) or {}
        sent += [{"name": name} for name in active.get("mcpServers", {})]
    return tuple(name for name in roster_names(sent) if name not in required)

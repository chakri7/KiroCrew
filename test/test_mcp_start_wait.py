"""Which MCP servers a session's first prompt waits for (``acp/mcp_start_wait.py``).

When a person is waiting on a start, the prompt waits for Kiro Crew's own servers
and, briefly, for the ones Kiro Crew sent on the session; it does not wait for the
agent's own (the user's) servers, as Kiro CLI does not. A start nobody is waiting
on (a sub-agent child, an unattended run) keeps the brief wait for those too.
"""

from __future__ import annotations

import asyncio

import pytest

from kiro_crew.acp import runtime_session_load
from kiro_crew.acp.mcp_session_report import McpSessionReport, required_managed_servers
from kiro_crew.acp.mcp_start_wait import first_prompt_brief_wait
from kiro_crew.acp.runtime import AcpRuntime
from kiro_crew.acp.session_handle import AcpSessionHandle
from kiro_crew.acp.session_handle_runtime_contract import AcpRuntimeDead, AcpRuntimeError
from kiro_crew.acp.types import ACP_BACKEND_KAS, METHOD_KAS_MCP_STATUS, JsonRpcMessage
from kiro_crew.mcp_cleanup import KIROCREW_BIN_MCP_SERVERS
from kiro_crew.start_priority import StartPriority
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS, async_wait_until, finished

FOREGROUND, BACKGROUND = StartPriority.FOREGROUND, StartPriority.BACKGROUND

_SID = "sess-start-wait"
#: An agent whose wire id is its own name (``kas_wire_agent_id``).
_AGENT = "worker"
_CORE = KIROCREW_BIN_MCP_SERVERS[0]
_CLIENT = {"kiro": {"resource": {"source": {"origin": "client"}}}}
#: The start as sent: Kiro Crew's own server and a Connection's bridge on the
#: session, and the user's two servers in the active agent's own block.
_PARAMS = {
    "sessionId": _SID,
    "mcpServers": [{"name": _CORE}, {"name": "bridge"}],
    "_meta": {
        "kiro": {
            "customAgents": [
                {"id": _AGENT, "mcpServers": {"users-fast": {}, "users-hung": {}}},
                {"id": "other", "mcpServers": {"not-this-agents": {}}},
            ]
        }
    },
}


class TestTheRule:
    def test_the_params_fixture_reads_as_the_runtime_reads_it(self):
        # Guards the fixture: if the wire shape of the agents batch moves, every
        # case below would pass for the wrong reason.
        assert required_managed_servers(_PARAMS, _AGENT) == (_CORE,)
        assert first_prompt_brief_wait(_PARAMS, _AGENT, (_CORE,), BACKGROUND) == (
            "bridge",
            "users-fast",
            "users-hung",
        )

    def test_a_person_waiting_does_not_wait_for_the_agents_own_servers(self):
        assert first_prompt_brief_wait(_PARAMS, _AGENT, (_CORE,), FOREGROUND) == ("bridge",)

    def test_a_start_nobody_waits_on_keeps_the_brief_wait_for_them(self):
        brief = first_prompt_brief_wait(_PARAMS, _AGENT, (_CORE,), BACKGROUND)

        assert {"users-fast", "users-hung"} <= set(brief)

    @pytest.mark.parametrize("priority", [FOREGROUND, BACKGROUND])
    def test_a_required_server_is_never_in_the_brief_set(self, priority):
        assert _CORE not in first_prompt_brief_wait(_PARAMS, _AGENT, (_CORE,), priority)

    @pytest.mark.parametrize("priority", [FOREGROUND, BACKGROUND])
    def test_another_agents_servers_are_not_this_starts(self, priority):
        assert "not-this-agents" not in first_prompt_brief_wait(_PARAMS, _AGENT, (), priority)

    def test_a_start_that_sent_nothing_waits_for_nothing(self):
        assert first_prompt_brief_wait({}, _AGENT, (), FOREGROUND) == ()
        assert first_prompt_brief_wait({}, _AGENT, (), BACKGROUND) == ()


def _status(**states: str) -> JsonRpcMessage:
    servers = []
    for name, status in states.items():
        entry = {"name": name, "status": status, "_meta": _CLIENT}
        if status == "connected":
            entry["tools"] = [{"name": "ping", "description": "probe", "disabled": False}]
        servers.append(entry)
    return JsonRpcMessage(
        method=METHOD_KAS_MCP_STATUS, params={"sessionId": _SID, "servers": servers}
    )


def _deliver(handle: AcpSessionHandle, frame: JsonRpcMessage | None) -> None:
    """Hand *frame* to the session as the runtime's reader does: report, then queue."""
    if frame is not None:
        handle._mcp_report.fold_frame(frame)
    handle._queue.put_nowait(frame)


def _handle(*frames: JsonRpcMessage | None) -> AcpSessionHandle:
    handle = AcpSessionHandle.__new__(AcpSessionHandle)
    handle._session_id = _SID
    handle._mcp_report = McpSessionReport(folded_on_arrival=True)
    handle._mcp_report.begin_session([])
    handle._queue = asyncio.Queue()
    for frame in frames:
        _deliver(handle, frame)
    handle._apply_init_notification = lambda *_a, **_k: None  # type: ignore[method-assign]
    return handle


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_session_load, "_declared_governed", lambda: False)
    monkeypatch.setattr(runtime_session_load, "MOUNTED_MCP_WAIT_SECS", FALLBACK_TIMEOUT_SECS)
    return AcpRuntime(work_dir=tmp_path / "ws", sandbox_mode="off", acp_backend=ACP_BACKEND_KAS)


def _start(runtime: AcpRuntime, handle: AcpSessionHandle, priority: StartPriority):
    return runtime._wait_managed_mcp(handle, _PARAMS, _AGENT, FALLBACK_TIMEOUT_SECS, 0, priority)


def _parked(handle: AcpSessionHandle) -> bool:
    return handle._queue.empty() and bool(handle._queue._getters)


class TestTheStart:
    @pytest.mark.asyncio
    async def test_a_person_waiting_starts_with_the_users_servers_still_connecting(self, runtime):
        """Ours and the bridge are up; the user's two are not, one of them never will be."""
        handle = _handle(
            _status(
                **{
                    _CORE: "connected",
                    "bridge": "connected",
                    "users-fast": "connecting",
                    "users-hung": "connecting",
                }
            )
        )

        await finished(_start(runtime, handle, FOREGROUND), "foreground start")

        assert handle._mcp_report.not_ready() == {"connecting": ("users-fast", "users-hung")}

    @pytest.mark.asyncio
    async def test_a_person_waiting_still_waits_for_kiro_crews_own_first(self, runtime):
        handle = _handle(_status(**{_CORE: "connecting", "bridge": "connected"}))
        task = asyncio.ensure_future(_start(runtime, handle, FOREGROUND))
        try:
            await async_wait_until(lambda: _parked(handle))
            assert not task.done()
            _deliver(handle, _status(**{_CORE: "connected", "bridge": "connected"}))
            await finished(task, "foreground start")
        finally:
            task.cancel()

    @pytest.mark.asyncio
    async def test_a_person_waiting_still_waits_briefly_for_a_session_sent_server(self, runtime):
        handle = _handle(_status(**{_CORE: "connected", "bridge": "connecting"}))
        task = asyncio.ensure_future(_start(runtime, handle, FOREGROUND))
        try:
            await async_wait_until(lambda: _parked(handle))
            assert not task.done()
            _deliver(handle, _status(**{_CORE: "connected", "bridge": "connected"}))
            await finished(task, "foreground start")
        finally:
            task.cancel()

    @pytest.mark.asyncio
    async def test_a_session_sent_server_that_never_answers_costs_the_cap_and_no_more(
        self, runtime, monkeypatch
    ):
        monkeypatch.setattr(runtime_session_load, "MOUNTED_MCP_WAIT_SECS", 0)
        handle = _handle(_status(**{_CORE: "connected", "bridge": "connecting"}))

        await finished(_start(runtime, handle, FOREGROUND), "foreground start")

        assert handle._mcp_report.not_ready() == {"connecting": ("bridge",)}

    @pytest.mark.asyncio
    async def test_a_start_nobody_waits_on_still_waits_for_the_users_servers(self, runtime):
        handle = _handle(
            _status(**{_CORE: "connected", "bridge": "connected", "users-fast": "connecting"})
        )
        task = asyncio.ensure_future(_start(runtime, handle, BACKGROUND))
        try:
            await async_wait_until(lambda: _parked(handle))
            assert not task.done()
            _deliver(
                handle,
                _status(
                    **{
                        _CORE: "connected",
                        "bridge": "connected",
                        "users-fast": "connected",
                        "users-hung": "connected",
                    }
                ),
            )
            await finished(task, "background start")
        finally:
            task.cancel()

    @pytest.mark.asyncio
    async def test_the_default_is_the_start_nobody_waits_on(self, runtime):
        """A caller that passes no priority (a child, a re-sent server set) is unchanged."""
        handle = _handle(
            _status(**{_CORE: "connected", "bridge": "connected", "users-fast": "connecting"})
        )
        task = asyncio.ensure_future(
            runtime._wait_managed_mcp(handle, _PARAMS, _AGENT, FALLBACK_TIMEOUT_SECS, 0)
        )
        try:
            await async_wait_until(lambda: _parked(handle))
            assert not task.done()
        finally:
            task.cancel()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("priority", [FOREGROUND, BACKGROUND])
    async def test_a_failed_required_server_fails_the_start_either_way(self, runtime, priority):
        handle = _handle(_status(**{_CORE: "failed", "bridge": "connected"}))

        with pytest.raises(AcpRuntimeError, match=_CORE):
            await _start(runtime, handle, priority)

    @pytest.mark.asyncio
    async def test_runtime_death_during_the_brief_wait_still_raises(self, runtime):
        handle = _handle(_status(**{_CORE: "connected", "bridge": "connecting"}), None)

        with pytest.raises(AcpRuntimeDead):
            await _start(runtime, handle, FOREGROUND)

    @pytest.mark.asyncio
    async def test_a_late_server_is_in_the_report_once_it_connects(self, runtime):
        """The start did not wait for it; the report follows it, so the next turn is told."""
        handle = _handle(
            _status(**{_CORE: "connected", "bridge": "connected", "users-fast": "connecting"})
        )
        await finished(_start(runtime, handle, FOREGROUND), "foreground start")
        assert "users-fast" in handle._mcp_report.not_ready()["connecting"]

        _deliver(
            handle,
            _status(**{_CORE: "connected", "bridge": "connected", "users-fast": "connected"}),
        )

        assert "users-fast" in handle._mcp_report.payload()["ready"]
        assert "users-fast" not in handle._mcp_report.not_ready().get("connecting", ())

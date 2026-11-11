"""Which MCP servers a session's first prompt waits for (``acp/mcp_start_wait.py``).

When a person is waiting on a start, the prompt waits for Kiro Crew's own servers
and, briefly, for the ones Kiro Crew sent on the session; the start does not wait
for the agent's own (the user's) servers, as Kiro CLI does not. A start nobody is
waiting on (a sub-agent child, an unattended run) keeps the brief wait for those
too.

The servers a start skipped are left to its first turn. A turn whose build read
the session's report names them to the model and goes out; any other first turn
(``kirocrew chat``, a side question) holds its prompt for them as the start did
before.
"""

from __future__ import annotations

import asyncio

import pytest

from kiro_crew.acp import runtime_session_load, session_handle_mcp
from kiro_crew.acp.mcp_session_report import McpSessionReport, required_managed_servers
from kiro_crew.acp.mcp_start_wait import first_prompt_brief_wait, left_to_first_turn
from kiro_crew.acp.runtime import AcpRuntime
from kiro_crew.acp.session_handle import AcpSessionHandle
from kiro_crew.acp.session_handle_mcp import hold_untold_first_turn
from kiro_crew.acp.session_handle_runtime_contract import AcpRuntimeDead, AcpRuntimeError
from kiro_crew.acp.types import ACP_BACKEND_KAS, METHOD_KAS_MCP_STATUS, JsonRpcMessage
from kiro_crew.agent_sdk.mcp_problems import TurnMcpStatesReader, session_mcp_states
from kiro_crew.mcp_cleanup import KIROCREW_BIN_MCP_SERVERS
from kiro_crew.start_priority import StartPriority
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS, async_wait_until, finished

FOREGROUND, BACKGROUND = StartPriority.FOREGROUND, StartPriority.BACKGROUND

_SID = "sess-start-wait"
#: An agent whose wire id is its own name (``kas_wire_agent_id``).
_AGENT = "worker"
_CORE = KIROCREW_BIN_MCP_SERVERS[0]
_CLIENT = {"kiro": {"resource": {"source": {"origin": "client"}}}}
#: The user's servers, which stay in the active agent's own block.
_USERS = ("users-fast", "users-hung")
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

    def test_a_person_waiting_leaves_the_agents_own_servers_to_the_first_turn(self):
        assert left_to_first_turn(_PARAMS, _AGENT, (_CORE,), FOREGROUND) == _USERS

    def test_a_start_nobody_waits_on_leaves_nothing_to_the_first_turn(self):
        assert left_to_first_turn(_PARAMS, _AGENT, (_CORE,), BACKGROUND) == ()

    @pytest.mark.parametrize("priority", [FOREGROUND, BACKGROUND])
    def test_every_own_server_is_waited_for_by_the_start_or_left_to_the_turn(self, priority):
        """No priority leaves a server of the agent's with neither."""
        waited = first_prompt_brief_wait(_PARAMS, _AGENT, (_CORE,), priority)
        left = left_to_first_turn(_PARAMS, _AGENT, (_CORE,), priority)

        assert set(_USERS) <= {*waited, *left}
        assert not set(waited) & set(left)

    def test_a_server_the_start_waited_for_is_not_left(self):
        """The agent's block and the session both carry it: the start waited briefly."""
        params = {**_PARAMS, "mcpServers": [{"name": _CORE}, {"name": "users-fast"}]}

        assert left_to_first_turn(params, _AGENT, (_CORE,), FOREGROUND) == ("users-hung",)


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


_OURS_UP = {_CORE: "connected", "bridge": "connected"}
_USERS_STARTING = {**_OURS_UP, "users-fast": "connecting", "users-hung": "connecting"}
_ALL_UP = {**_OURS_UP, "users-fast": "connected", "users-hung": "connected"}


async def _started(runtime: AcpRuntime) -> AcpSessionHandle:
    """A session a person waited on, started with the user's two servers still connecting."""
    handle = _handle(_status(**_USERS_STARTING))
    await finished(_start(runtime, handle, FOREGROUND), "foreground start")
    return handle


class _Sessions:
    """A session manager that holds one session, under ``dashboard:tab-1``."""

    def __init__(self, handle: AcpSessionHandle) -> None:
        self._handle = handle

    def get_provider(self, key: str) -> AcpSessionHandle | None:
        return self._handle if key == "dashboard:tab-1" else None


#: What a report holds before any block was built for its session.
_UNTOLD = ("", frozenset())
#: A block's mark, as ``context_assembly.mcp_servers`` makes one per block.
_MARK = "<<<BEGIN_MCP_SERVER_STATES_0a1b2c3d>>>"
_PROMPT_WITH_THE_BLOCK = f"[MCP SERVERS] ...\n{_MARK}\nstill starting: users-fast\n...\nping"


def _holding(handle: AcpSessionHandle) -> bool:
    """The hold made its signal, so it is parked on it: nothing awaits in between."""
    return handle._mcp_report.moved is not None


def _queued(handle: AcpSessionHandle) -> list:
    frames = []
    while not handle._queue.empty():
        frames.append(handle._queue.get_nowait())
    return frames


class TestTheFirstTurn:
    """What the first turn does with the servers its start skipped (``hold_untold_first_turn``)."""

    @pytest.fixture(autouse=True)
    def cap(self, monkeypatch):
        monkeypatch.setattr(session_handle_mcp, "MOUNTED_MCP_WAIT_SECS", FALLBACK_TIMEOUT_SECS)

    @pytest.mark.asyncio
    async def test_a_start_a_person_waited_on_leaves_the_users_servers_to_it(self, runtime):
        handle = await _started(runtime)

        assert handle._mcp_report.left_to_turn == _USERS
        assert handle._mcp_report.turn_told == _UNTOLD

    @pytest.mark.asyncio
    async def test_a_start_nobody_waited_on_leaves_nothing(self, runtime):
        handle = _handle(_status(**_ALL_UP))

        await finished(_start(runtime, handle, BACKGROUND), "background start")

        assert handle._mcp_report.left_to_turn == ()
        await finished(hold_untold_first_turn(handle), "first turn")

    @pytest.mark.asyncio
    async def test_a_turn_that_was_not_told_holds_until_they_report(self, runtime):
        handle = await _started(runtime)
        hold = asyncio.ensure_future(hold_untold_first_turn(handle))
        try:
            await async_wait_until(lambda: _holding(handle))
            assert not hold.done()
            _deliver(handle, _status(**{**_USERS_STARTING, "users-fast": "connected"}))
            await async_wait_until(lambda: not handle._mcp_report.moved.is_set())
            assert not hold.done()
            _deliver(handle, _status(**_ALL_UP))
            await finished(hold, "untold first turn")
        finally:
            hold.cancel()

        assert handle._mcp_report.left_to_turn == ()

    @pytest.mark.asyncio
    async def test_a_prompt_that_carries_the_block_goes_out_while_they_still_connect(self, runtime):
        handle = await _started(runtime)

        TurnMcpStatesReader(_Sessions(handle)).told("dashboard:tab-1", _MARK, _USERS)
        await finished(hold_untold_first_turn(handle, _PROMPT_WITH_THE_BLOCK), "told first turn")

        assert handle._mcp_report.not_ready() == {"connecting": _USERS}
        assert handle._mcp_report.left_to_turn == ()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt",
        ["ping", "", _PROMPT_WITH_THE_BLOCK.replace("0a1b2c3d", "ffffffff")],
        ids=["a-bare-prompt", "a-command", "another-block"],
    )
    async def test_a_block_that_was_built_exempts_only_the_prompt_that_carries_it(
        self, runtime, prompt
    ):
        """A build can be stopped before it is sent; the next prompt is then not told."""
        handle = await _started(runtime)
        TurnMcpStatesReader(_Sessions(handle)).told("dashboard:tab-1", _MARK, _USERS)

        hold = asyncio.ensure_future(hold_untold_first_turn(handle, prompt))
        try:
            await async_wait_until(lambda: _holding(handle))
            assert not hold.done()
        finally:
            hold.cancel()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("named", [(), ("users-fast",), ("users-fast", "bridge")])
    async def test_a_block_that_leaves_out_a_server_the_start_skipped_tells_no_turn(
        self, runtime, named
    ):
        """Whatever kept it out of the row: screening, the row's cap, a changed name."""
        handle = await _started(runtime)

        TurnMcpStatesReader(_Sessions(handle)).told("dashboard:tab-1", _MARK, named)

        hold = asyncio.ensure_future(hold_untold_first_turn(handle, _PROMPT_WITH_THE_BLOCK))
        try:
            await async_wait_until(lambda: _holding(handle))
            assert not hold.done()
        finally:
            hold.cancel()

    @pytest.mark.asyncio
    async def test_a_block_need_not_name_a_skipped_server_that_has_reported(self, runtime):
        handle = await _started(runtime)
        _deliver(handle, _status(**{**_USERS_STARTING, "users-hung": "failed"}))

        TurnMcpStatesReader(_Sessions(handle)).told("dashboard:tab-1", _MARK, ("users-fast",))

        await finished(hold_untold_first_turn(handle, _PROMPT_WITH_THE_BLOCK), "told first turn")

    @pytest.mark.asyncio
    async def test_a_server_that_goes_back_to_connecting_after_the_build_holds_the_prompt(
        self, runtime
    ):
        """The block was built while it was connected, so it does not name it. The
        decision is made when the prompt is sent, on what is unreported then."""
        handle = await _started(runtime)
        _deliver(handle, _status(**{**_USERS_STARTING, "users-fast": "connected"}))
        TurnMcpStatesReader(_Sessions(handle)).told("dashboard:tab-1", _MARK, ("users-hung",))
        assert handle._mcp_report.turn_is_told(_PROMPT_WITH_THE_BLOCK)

        _deliver(handle, _status(**_USERS_STARTING))

        assert handle._mcp_report.unreported(_USERS) == _USERS
        hold = asyncio.ensure_future(hold_untold_first_turn(handle, _PROMPT_WITH_THE_BLOCK))
        try:
            await async_wait_until(lambda: _holding(handle))
            assert not hold.done()
        finally:
            hold.cancel()

    @pytest.mark.asyncio
    async def test_the_reader_tells_only_the_session_its_key_names(self, runtime):
        handle = await _started(runtime)
        reader = TurnMcpStatesReader(_Sessions(handle))

        reader.told("dashboard:other", _MARK, _USERS)

        assert reader("dashboard:other") is None
        assert handle._mcp_report.turn_told == _UNTOLD

    @pytest.mark.asyncio
    async def test_a_reading_alone_tells_no_turn(self, runtime):
        """A block can be read for and still not be sent, so only ``told`` counts."""
        handle = await _started(runtime)

        assert TurnMcpStatesReader(_Sessions(handle))("dashboard:tab-1")["connecting"] == _USERS
        assert session_mcp_states(handle) is not None
        hold = asyncio.ensure_future(hold_untold_first_turn(handle))
        try:
            await async_wait_until(lambda: _holding(handle))
            assert not hold.done()
        finally:
            hold.cancel()

    @pytest.mark.asyncio
    async def test_servers_that_reported_since_the_start_hold_no_turn(self, runtime):
        handle = await _started(runtime)
        _deliver(handle, _status(**{**_OURS_UP, "users-fast": "connected", "users-hung": "failed"}))

        await finished(hold_untold_first_turn(handle), "untold first turn")

        assert handle._mcp_report.turn_told == _UNTOLD

    @pytest.mark.asyncio
    async def test_a_server_that_never_answers_costs_an_untold_turn_the_cap_and_no_more(
        self, runtime, monkeypatch
    ):
        handle = await _started(runtime)
        monkeypatch.setattr(session_handle_mcp, "MOUNTED_MCP_WAIT_SECS", 0)

        await finished(hold_untold_first_turn(handle), "untold first turn")

        assert handle._mcp_report.not_ready() == {"connecting": _USERS}
        assert handle._mcp_report.left_to_turn == ()

    @pytest.mark.asyncio
    async def test_only_the_first_turn_holds(self, runtime, monkeypatch):
        handle = await _started(runtime)
        monkeypatch.setattr(session_handle_mcp, "MOUNTED_MCP_WAIT_SECS", 0)
        await finished(hold_untold_first_turn(handle), "first turn")
        monkeypatch.setattr(session_handle_mcp, "MOUNTED_MCP_WAIT_SECS", FALLBACK_TIMEOUT_SECS)

        await finished(hold_untold_first_turn(handle), "second turn")

    @pytest.mark.asyncio
    async def test_a_hold_that_was_cancelled_holds_the_next_turn(self, runtime):
        handle = await _started(runtime)
        hold = asyncio.ensure_future(hold_untold_first_turn(handle))
        await async_wait_until(lambda: _holding(handle))
        hold.cancel()
        await asyncio.gather(hold, return_exceptions=True)

        assert handle._mcp_report.left_to_turn == _USERS

    @pytest.mark.asyncio
    async def test_the_hold_takes_nothing_off_the_sessions_queue(self, runtime):
        """What sits there between turns is the turn's own drain to keep, answer or drop:
        a command's awaited response, a stranded permission request, the runtime's death."""
        handle = await _started(runtime)
        assert _queued(handle) == []
        answer = JsonRpcMessage(id=7, result={})
        request = JsonRpcMessage(id=8, method="session/request_permission", params={})
        for frame in (answer, request):
            handle._queue.put_nowait(frame)
        hold = asyncio.ensure_future(hold_untold_first_turn(handle))
        try:
            await async_wait_until(lambda: _holding(handle))
            up = _status(**_ALL_UP)
            _deliver(handle, up)
            handle._queue.put_nowait(None)
            await finished(hold, "untold first turn")
        finally:
            hold.cancel()

        assert _queued(handle) == [answer, request, up, None]

    @pytest.mark.asyncio
    async def test_a_new_session_attempt_starts_untold_with_nothing_left(self, runtime):
        handle = await _started(runtime)
        handle._mcp_report.note_turn_told(_MARK, _USERS)

        handle._mcp_report.begin_session([])

        assert handle._mcp_report.left_to_turn == ()
        assert handle._mcp_report.turn_told == _UNTOLD

"""Resume on a turn whose message never reached the agent re-sends that message.

A ``memory_unavailable`` error ends a turn while its context is still being
prepared, so the agent's session never holds the user's message. The failing
turn stamps that message's ``mid`` on its error row (``unsentMid``), and Resume
then queues the message itself, not the generic resume body.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_runner_coverage_helpers import _quiet_sel, _runner_state, _settle

from kiro_crew.dashboard import chat_handlers, chat_runner
from kiro_crew.dashboard.chat_handlers import _MANUAL_RESUME_MSG, api_chat_slot_continue
from kiro_crew.dashboard.chat_runner import MIRROR_ECHO_OWED_META_KEY, _batch_recovery_echo_owed
from kiro_crew.dashboard.chat_utils import (
    SYNTHETIC_RECOVERY_KIND,
    RecoveryPayload,
    is_synthetic_payload_item,
)
from kiro_crew.dashboard.state import DashboardState, _ChatSlot, row_mid
from kiro_crew.memory_stores import UnknownMemoryStore

_POOL_BUSY = (
    "memory_unavailable: the memory pool was busy and this turn's context could "
    "not be prepared in time; nothing was sent -- try again in a moment"
)


def _row(role: str, text: str, mid: str = "", **meta) -> dict:
    if mid:
        meta["mid"] = mid
    return {"role": role, "content": text, "meta": meta}


def _mem_error(unsent_mid: str = "") -> dict:
    meta: dict = {"code": "memory_unavailable"}
    if unsent_mid:
        meta["unsentMid"] = unsent_mid
    return {"role": "error", "content": _POOL_BUSY, "meta": meta}


def unsent_user_message(rows: list[dict]) -> str | None:
    row = chat_handlers.unsent_user_row(rows)
    return None if row is None else row["content"]


class TestUnsentUserMessage:
    def test_the_stamped_message_is_returned(self):
        rows = [_row("user", "old", "m0"), _row("user", "leave it on", "m1"), _mem_error("m1")]
        assert unsent_user_message(rows) == "leave it on"

    def test_preparation_rows_before_the_error_are_never_consulted(self):
        rows = [
            _row("user", "@standup leave it on", "m1"),
            _row("system", "Prompt standup loaded", "m2"),
            _row("system", "Loaded skill review", "m3", kind="skill_load"),
            _mem_error("m1"),
        ]
        assert unsent_user_message(rows) == "@standup leave it on"

    def test_the_turn_end_marker_after_the_error_is_passed_over(self):
        rows = [_row("user", "leave it on", "m1"), _mem_error("m1"), _row("done", "")]
        assert unsent_user_message(rows) == "leave it on"

    def test_a_second_failure_after_a_resend_returns_the_replay(self):
        rows = [
            _row("user", "leave it on", "m1"),
            _mem_error("m1"),
            _row("inject", "leave it on", "m2", injectKind="user_replay"),
            _mem_error("m2"),
        ]
        assert unsent_user_message(rows) == "leave it on"

    def test_an_unstamped_error_keeps_the_generic_resume(self):
        assert unsent_user_message([_row("user", "leave it on", "m1"), _mem_error()]) is None

    @pytest.mark.parametrize("role", ["assistant", "tool", "user", "inject", "nudge", "subagent"])
    def test_a_turn_row_after_the_error_keeps_the_generic_resume(self, role):
        rows = [_row("user", "leave it on", "m1"), _mem_error("m1"), _row(role, "later", "m2")]
        assert unsent_user_message(rows) is None

    def test_another_failure_kind_keeps_the_generic_resume(self):
        rows = [_row("user", "do it", "m1"), _row("error", "connection lost", unsentMid="m1")]
        assert unsent_user_message(rows) is None

    def test_a_stamp_naming_no_row_keeps_the_generic_resume(self):
        assert unsent_user_message([_row("user", "do it", "m1"), _mem_error("gone")]) is None

    @pytest.mark.parametrize(
        "named",
        [
            pytest.param(_row("assistant", "a reply", "m1"), id="assistant"),
            pytest.param(_row("inject", "cron says", "m1", injectKind="cron"), id="cron-inject"),
            pytest.param(_row("user", "   ", "m1"), id="blank"),
        ],
    )
    def test_a_stamp_naming_anything_but_a_users_words_keeps_the_generic_resume(self, named):
        assert unsent_user_message([named, _mem_error("m1")]) is None


def _make_app(state: DashboardState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/continue", api_chat_slot_continue)
    return app


def _mock_state(slot: _ChatSlot) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._slots = {slot.key: slot}
    state.push_slots_update = MagicMock()
    state.broadcast_ws = MagicMock()
    state.subagents = None
    return state


@pytest.fixture
def _patched():
    mock_sel = MagicMock()
    with (
        patch("kiro_crew.dashboard.chat_handlers.sel", return_value=mock_sel),
        patch(
            "kiro_crew.dashboard.chat_handlers._start_next_queued_turn",
            AsyncMock(return_value=True),
        ),
    ):
        yield


async def _press_resume(slot: _ChatSlot) -> int:
    async with TestClient(TestServer(_make_app(_mock_state(slot)))) as client:
        resp = await client.post("/api/chat/slots/s/continue")
        return resp.status


def _fail_stamped(slot: _ChatSlot, sent: dict, **extra) -> None:
    slot.append(
        "error",
        _POOL_BUSY,
        "msg msg-err",
        meta={"code": "memory_unavailable", "unsentMid": row_mid(sent), **extra},
    )


class TestResumeAfterPoolBusy:
    @pytest.mark.asyncio
    async def test_resume_after_a_real_pool_busy_turn_resends_the_message(self, _patched, tmp_path):
        state, _client = _runner_state(tmp_path)
        slot = _ChatSlot("s")
        slot._titled = True
        slot.append("user", "Leave the watchers on", "msg msg-u")
        slot._empty_response_retries = 2
        refusal = UnknownMemoryStore("the memory pool was busy")
        with (
            _quiet_sel(),
            patch.object(chat_runner, "resolve_agent_bindings", side_effect=refusal),
        ):
            await chat_runner._run_chat(
                state, slot, "Leave the watchers on", _directive_user_origin=True
            )
        await _settle(slot)

        assert await _press_resume(slot) == 200

        entry = slot._queue[0]
        assert entry["content"] == "Leave the watchers on"
        assert entry["kind"] == SYNTHETIC_RECOVERY_KIND
        assert entry["payload"] == RecoveryPayload.ORIGINAL
        assert is_synthetic_payload_item(entry) is False
        assert _batch_recovery_echo_owed([entry]) is True

    @pytest.mark.asyncio
    async def test_the_resend_owes_the_linked_channel_the_echo_the_failed_turn_never_posted(
        self, _patched
    ):
        slot = _ChatSlot("s")
        sent = slot.append("user", "Leave the watchers on", "msg msg-u")
        _fail_stamped(slot, sent, **{MIRROR_ECHO_OWED_META_KEY: True})

        assert await _press_resume(slot) == 200

        entry = slot._queue[0]
        assert entry["meta"][MIRROR_ECHO_OWED_META_KEY] is True
        assert _batch_recovery_echo_owed([entry]) is True

    @pytest.mark.asyncio
    async def test_a_resend_whose_error_owes_no_echo_carries_no_echo_stamp(self, _patched):
        slot = _ChatSlot("s")
        sent = slot.append("user", "Leave the watchers on", "msg msg-u")
        _fail_stamped(slot, sent)

        assert await _press_resume(slot) == 200

        entry = slot._queue[0]
        assert MIRROR_ECHO_OWED_META_KEY not in entry["meta"]
        assert _batch_recovery_echo_owed([entry]) is False

    @pytest.mark.asyncio
    async def test_prompt_chip_and_skill_rows_before_the_error_still_resend(self, _patched):
        slot = _ChatSlot("s")
        sent = slot.append("user", "@standup $review go", "msg msg-u")
        slot.append("system", "Prompt standup loaded", "msg msg-sys")
        slot.append("system", "Loaded skill review", "msg msg-sys", meta={"kind": "skill_load"})
        _fail_stamped(slot, sent)

        assert await _press_resume(slot) == 200

        assert slot._queue[0]["content"] == "@standup $review go"
        assert slot._queue[0]["payload"] == RecoveryPayload.ORIGINAL

    @pytest.mark.asyncio
    async def test_the_resend_carries_the_attachments_of_the_unsent_message(self, _patched):
        slot = _ChatSlot("s")
        sent = slot.append(
            "user", "Review these", "msg msg-u", meta={"files": ["/w/a.png"], "dirs": ["/w/src"]}
        )
        _fail_stamped(slot, sent)

        assert await _press_resume(slot) == 200

        meta = slot._queue[0]["meta"]
        assert meta["files"] == ["/w/a.png"]
        assert meta["dirs"] == ["/w/src"]

    @pytest.mark.asyncio
    async def test_resume_resends_on_the_live_tail_a_finished_turn_leaves(self, _patched):
        slot = _ChatSlot("s")
        sent = slot.append("user", "Leave the watchers on", "msg msg-u")
        _fail_stamped(slot, sent)
        slot.append("done", "", "done")

        assert await _press_resume(slot) == 200

        assert slot._queue[0]["content"] == "Leave the watchers on"
        assert slot._queue[0]["payload"] == RecoveryPayload.ORIGINAL

    @pytest.mark.asyncio
    async def test_a_second_failure_after_a_resend_resends_again(self, _patched):
        slot = _ChatSlot("s")
        sent = slot.append("user", "Leave the watchers on", "msg msg-u")
        _fail_stamped(slot, sent)
        replay = slot.append(
            "inject", "Leave the watchers on", "msg msg-inj", meta={"injectKind": "user_replay"}
        )
        _fail_stamped(slot, replay)

        assert await _press_resume(slot) == 200

        assert slot._queue[0]["content"] == "Leave the watchers on"
        assert slot._queue[0]["payload"] == RecoveryPayload.ORIGINAL

    @pytest.mark.asyncio
    async def test_an_unstamped_error_keeps_the_generic_resume_body(self, _patched):
        slot = _ChatSlot("s")
        slot.append("user", "Leave the watchers on", "msg msg-u")
        slot.append("error", _POOL_BUSY, "msg msg-err", meta={"code": "memory_unavailable"})

        assert await _press_resume(slot) == 200

        assert slot._queue[0]["content"] == _MANUAL_RESUME_MSG
        assert is_synthetic_payload_item(slot._queue[0]) is True

    @pytest.mark.asyncio
    async def test_output_after_the_error_keeps_the_generic_resume_body(self, _patched):
        slot = _ChatSlot("s")
        sent = slot.append("user", "Leave the watchers on", "msg msg-u")
        _fail_stamped(slot, sent)
        slot.append("assistant", "Working on it", "msg msg-a")
        slot.append("error", "connection lost", "msg msg-err")

        assert await _press_resume(slot) == 200

        assert slot._queue[0]["content"] == _MANUAL_RESUME_MSG
        assert is_synthetic_payload_item(slot._queue[0]) is True

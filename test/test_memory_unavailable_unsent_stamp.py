"""A ``memory_unavailable`` error row names the message its turn never sent.

The stamp (``meta.unsentMid``) is what lets Resume resend that message, and the
resend runs as the presser's own turn, so only a person's own dashboard prompt,
or a verbatim Resume replay of one, the agent did not consume may carry it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from chat_runner_coverage_helpers import _quiet_sel, _runner_state, _settle, _slot

from kiro_crew.dashboard import chat_runner
from kiro_crew.dashboard.chat_runner import MIRROR_ECHO_OWED_META_KEY
from kiro_crew.dashboard.chat_turn import turn_failures
from kiro_crew.dashboard.state import row_mid
from kiro_crew.memory_stores import UnknownMemoryStore


async def _fail_turn(tmp_path, *, row_role: str = "user", row_meta=None, **run_kwargs) -> tuple:
    """Run one turn that ends ``memory_unavailable``; return (error row, sent row)."""
    state, _client = _runner_state(tmp_path)
    slot = _slot()
    sent = slot.append(row_role, "Leave the watchers on", "msg msg-u", meta=row_meta)
    slot._empty_response_retries = 2
    refusal = UnknownMemoryStore("the memory pool was busy")
    with (
        _quiet_sel(),
        patch.object(chat_runner, "resolve_agent_bindings", side_effect=refusal),
    ):
        await chat_runner._run_chat(state, slot, "Leave the watchers on", **run_kwargs)
    await _settle(slot)
    error = next(row for row in slot.messages if row["role"] == "error")
    assert error["meta"]["code"] == "memory_unavailable"
    return error, sent


@pytest.mark.asyncio
async def test_a_persons_own_dashboard_prompt_is_stamped(tmp_path):
    error, sent = await _fail_turn(tmp_path, _directive_user_origin=True)

    assert row_mid(sent)
    assert error["meta"]["unsentMid"] == row_mid(sent)


@pytest.mark.asyncio
async def test_a_verbatim_resume_replay_is_stamped_so_a_second_failure_resends(tmp_path):
    error, sent = await _fail_turn(
        tmp_path,
        row_role="inject",
        row_meta={"injectKind": "user_replay"},
        _directive_user_origin=True,
        _synthetic_recovery_turn=True,
    )

    assert error["meta"]["unsentMid"] == row_mid(sent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_kwargs",
    [
        pytest.param(
            {"_directive_user_origin": True, "_directive_channel_origin": True}, id="channel"
        ),
        pytest.param({"_directive_user_origin": False, "_card_origin": True}, id="card"),
        pytest.param({"_directive_user_origin": True, "_turn_actor": "app"}, id="app-actor"),
        pytest.param({"_directive_user_origin": True, "_turn_actor": "crew"}, id="crew-actor"),
        pytest.param({"_directive_user_origin": True, "_commands_off": True}, id="commands-off"),
        pytest.param(
            {"_directive_user_origin": True, "_turn_provenance_restored": True}, id="restored"
        ),
        pytest.param({"_directive_user_origin": False}, id="not-user-origin"),
        pytest.param(
            {
                "_directive_user_origin": True,
                "_synthetic_recovery_turn": True,
                "_synthetic_payload": True,
            },
            id="runner-written-recovery",
        ),
        pytest.param(
            {
                "_directive_user_origin": True,
                "_synthetic_recovery_turn": True,
                "_turn_actor": "cron",
            },
            id="cron-replay",
        ),
        pytest.param(
            {
                "_directive_user_origin": True,
                "_synthetic_recovery_turn": True,
                "_turn_provenance_restored": True,
            },
            id="restored-replay",
        ),
    ],
)
async def test_a_turn_that_is_not_the_persons_own_prompt_is_not_stamped(tmp_path, run_kwargs):
    error, _sent = await _fail_turn(tmp_path, **run_kwargs)

    assert "unsentMid" not in error["meta"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_kwargs",
    [
        pytest.param({"_directive_user_origin": True}, id="dashboard-prompt"),
        pytest.param(
            {
                "_directive_user_origin": True,
                "_synthetic_recovery_turn": True,
                "_recovery_echo_owed": True,
            },
            id="replay-still-owing-its-echo",
        ),
    ],
)
async def test_a_prompt_whose_echo_was_never_posted_stamps_that_it_is_owed(tmp_path, run_kwargs):
    error, sent = await _fail_turn(tmp_path, **run_kwargs)

    assert error["meta"]["unsentMid"] == row_mid(sent)
    assert error["meta"][MIRROR_ECHO_OWED_META_KEY] is True


@pytest.mark.asyncio
async def test_a_replay_that_owes_no_echo_is_resendable_without_the_echo_stamp(tmp_path):
    """The first attempt's echo was posted, so the resend must not quote it again."""
    error, sent = await _fail_turn(
        tmp_path,
        row_role="inject",
        row_meta={"injectKind": "user_replay"},
        _directive_user_origin=True,
        _synthetic_recovery_turn=True,
    )

    assert error["meta"]["unsentMid"] == row_mid(sent)
    assert MIRROR_ECHO_OWED_META_KEY not in error["meta"]


@pytest.mark.asyncio
async def test_a_prompt_whose_echo_was_already_posted_is_not_stamped_as_owing_it(tmp_path):
    real = chat_runner._resendable_unsent_mid

    def _after_the_echo(turn):
        turn._mirror_echo_posted = True
        return real(turn)

    with patch.object(chat_runner, "_resendable_unsent_mid", side_effect=_after_the_echo):
        error, sent = await _fail_turn(tmp_path, _directive_user_origin=True)

    assert error["meta"]["unsentMid"] == row_mid(sent)
    assert MIRROR_ECHO_OWED_META_KEY not in error["meta"]


def _bare_turn() -> SimpleNamespace:
    row = {"role": "user", "content": "x", "meta": {"mid": "m-1"}}
    return SimpleNamespace(
        _consumed_reported=False,
        _commands_off=False,
        _synthetic_recovery_turn=False,
        _synthetic_payload=False,
        _directive_user_origin=True,
        _directive_channel_origin=False,
        _crew_log_actor="user",
        _recovery_echo_owed=False,
        _turn_provenance_restored=False,
        _current_replay_message=row,
        slot=SimpleNamespace(_turn_generation=1, _handoff_generation=-1),
    )


def test_a_turn_the_agent_already_consumed_is_not_stamped():
    """``memory_unavailable`` is raised before dispatch, so no live turn reaches
    this with the flag set; the gate keeps the stamp honest if one ever does."""
    turn = _bare_turn()
    assert turn_failures._resendable_unsent_mid(turn) == "m-1"

    turn._consumed_reported = True

    assert turn_failures._resendable_unsent_mid(turn) is None


def test_a_turn_whose_prompt_stream_already_opened_is_not_stamped():
    """A binding check during the stream can raise ``memory_unavailable`` after the
    prompt reached the agent, before any consuming event; resending it would
    deliver the same prompt twice."""
    turn = _bare_turn()
    turn.slot._handoff_generation = turn.slot._turn_generation

    assert turn_failures._resendable_unsent_mid(turn) is None

"""Where a write of a carried fold lands once its slot may have moved on: decided on the lane.

A slot's newest unit carries the slot's fold state (``session/carried``), and the fold
replaces whatever the older units gave it with that state. A write that names an older
unit AFTER such a carry would read wrongly either way: its entry sits in a unit the carry
already replaced, so it is lost, or that unit is published as the unit that recorded
last, so an unpinned read folds it AFTER the carry and lays its whole history over the
state the carry already holds.

So every writer that appends a carried fold's entry into a unit it resolved before its
job runs decides the unit in that job: the ledger writers (``session_ledger.record_update``
and the Issue Radar ledger, through ``emit_surfaces.on_ledger_recorded`` and
``on_radar_recorded``), the work board's (``emit_surfaces.on_work_recorded``, whose unit
the report route resolves off the writer), the board relay's (``work_relay._queue_run``,
whose unit its consumer thread resolves) and the crew panel's
(``emit_surfaces.on_panel_published``). The job asks :func:`hand_to_carrier`, appends, and
publishes the unit order for the unit the entry landed in. It runs in the named unit's
bucket, and the writer holds one slot's units to one lane (``writer_groups``), which is
where the successor's carry is committed too (``emit_session._write_carries``, which calls
:func:`note_carried` before it publishes the carrier's order). So the carry lands wholly
before the job or wholly after it. Before, the job finds it and hands itself to the
slot's newest unit, which asks again. After, the carry reads the entry: the older unit's
tail moved since the carry's fold read it, so the carry folds again. Nothing compares a
clock, and nothing guesses which entry is whose: a ledger job hands its caller a
:class:`Receipt` naming the unit and seq the entry landed at.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Final

logger = logging.getLogger(__name__)

#: Most ``(data home, slot, fold)`` carriers remembered, the oldest forgotten first: the
#: bound ``writer_groups.UNIT_SLOTS_LIMIT`` puts on the units the writer keys by slot.
CARRIERS_LIMIT: Final[int] = 4096

#: ``(data home, slot, fold)`` -> the unit whose carry of that fold landed last in this
#: process. Written on the slot's lane by the carry, read there by the writer job.
_CARRIERS: "OrderedDict[tuple[str, str, str], str]" = OrderedDict()
_guard = threading.Lock()


class Receipt:
    """Where one ledger write landed, settled once by the writer job that made it.

    ``unit`` is the unit the entry is in (the slot's newest one when the job redirected
    it), ``seq`` its position there, and ``landed`` whether the append returned. Until it
    is settled, ``landed`` is False: an entry still queued or dropped is not in the file.
    Settled on the writer thread, read by the caller once :meth:`wait` (or a writer
    flush) returns.
    """

    __slots__ = ("unit", "seq", "landed", "_done", "_lock")

    def __init__(self, unit: str) -> None:
        self.unit = unit
        self.seq = 0
        self.landed = False
        self._done = threading.Event()
        self._lock = threading.Lock()

    def settle(self, unit: str, seq: int, *, landed: bool) -> None:
        """Record the outcome. Only the first call counts."""
        with self._lock:
            if self._done.is_set():
                return
            self.unit = unit
            self.seq = seq
            self.landed = landed
            self._done.set()

    @property
    def settled(self) -> bool:
        """Whether the job has said what became of the entry."""
        return self._done.is_set()

    def wait(self, timeout: "float | None") -> bool:
        """Wait at most *timeout* seconds for the outcome; True once it is settled."""
        return self._done.wait(timeout)


def note_carried(slot: str, unit: str, folds: "Iterable[str]") -> None:
    """Remember that *unit*'s carry of each of *folds* for *slot* has landed.

    Called by the carry on the slot's lane, right after its group commit and before the
    carrier's order is published, so a writer job on the same lane that runs after it
    finds it.
    """
    if not slot or not unit:
        return
    home = _home()
    with _guard:
        for fold in folds:
            key = (home, slot, fold)
            _CARRIERS[key] = unit
            _CARRIERS.move_to_end(key)
        while len(_CARRIERS) > CARRIERS_LIMIT:
            _CARRIERS.popitem(last=False)


def clear() -> None:
    """Forget every carrier. For ``emit.reset_caches``, which stands in for a restart."""
    with _guard:
        _CARRIERS.clear()


def carrier_after(slot: str, unit_id: str, *folds: str) -> str:
    """The slot's newest unit when *slot* moved past *unit_id* for any of *folds*, else ``""``.

    Moved past means a unit AFTER *unit_id* in the slot's succession carried one of *folds*
    (:func:`note_carried`), or *unit_id* is already closed ``superseded``: its successor
    carried every fold it held, and retention may collect it. ``""`` when *unit_id* is the
    slot's newest unit, is not in its succession (another spelling's unit), no later unit
    carried yet (a carry still to land reads the unit's entries when it does), or anything
    cannot be read: the write then goes where it always went.

    Cheap for the ordinary writer, which is the newest unit: the succession is cached and
    no log is opened. Exact on the slot's lane, where the writer job asks it; asked off
    the lane it is a first guess the job asks again. A carry an earlier process wrote is
    seen only through the close it wrote after it (``emit_session._close_superseded``).
    """
    if not slot or not unit_id:
        return ""
    try:
        from kiro_crew.crew_log import projection

        units = projection.slot_succession(slot)
        if unit_id not in units or units[-1] == unit_id:
            return ""
        later = units[units.index(unit_id) + 1 :]
        home = _home()
        with _guard:
            carriers = [_CARRIERS.get((home, slot, fold), "") for fold in folds]
        if any(carrier in later for carrier in carriers) or _closed_superseded(unit_id):
            return units[-1]
    except Exception:
        # Rendered to text: a kept traceback keeps this frame, and a crew log handle bound in it.
        from kiro_crew.crew_log.store import log_exception_text

        log_exception_text(
            logger,
            logging.DEBUG,
            "crew log %s: could not tell whether slot %s moved on",
            unit_id,
            slot,
        )
    return ""


@functools.lru_cache(maxsize=None)
def folds_fed_by(entry_type: str) -> "tuple[str, ...]":
    """Every carried fold an *entry_type* entry can move, by the fold registry's own sets.

    A ``work/recorded`` entry moves the board's ``work`` and ``workstreams`` folds, the
    work tree, the worker rows and ``status``; ``work/charged`` moves ``workstreams`` and
    ``status``. Read off each fold's ``affects`` (``_Fold.touched_by_type``) rather than
    listed here, so a fold that starts reading the type is asked about the day it does.
    Cached: the registry is built at import and never changes.
    """
    from kiro_crew.crew_log import projection

    return tuple(
        name
        for name in sorted(projection.CARRIED_FOLD_NAMES)
        if projection._FOLDS[name].touched_by_type(entry_type)
    )


def hand_to_carrier(
    slot: str, unit: str, folds: "Iterable[str]", land: "Callable[[str], None]", what: str
) -> bool:
    """Hand ``land(newer)`` to *slot*'s newest unit when it carried one of *folds* past *unit*.

    From the writer job that was about to append into *unit*, on the slot's lane (the
    writer thread, or the caller's under the writer's inline claim; the module docstring
    says why that lane decides). True when the job was handed on: the caller then appends
    nothing, and the follow-on job, in the newer unit's own bucket, runs *land* there,
    which asks again. False when the entry belongs in *unit* (:func:`carrier_after`
    answered ``""``, or *slot* is empty). The newer unit's slot is noted for the writer
    first, so its bucket is held to the same lane.
    """
    newer = carrier_after(slot, unit, *folds)
    if not newer or newer == unit:
        return False
    from kiro_crew.crew_log import writer_groups
    from kiro_crew.crew_log.emit_runtime import _hand_over
    from kiro_crew.crew_log.writer import WriteJob

    if not writer_groups.known(newer):
        writer_groups.note_unit_slot(newer, slot)
    _hand_over(newer, WriteJob.follow_on(functools.partial(land, newer), what))
    return True


def _closed_superseded(unit_id: str) -> bool:
    """Whether *unit_id*'s newest lifecycle entry is a ``superseded`` close. A tail read."""
    from kiro_crew.crew_log.entry_types import SUPERSEDED_CLOSE_REASON
    from kiro_crew.crew_log.schema import KIND_SESSION
    from kiro_crew.crew_log.store import _newest_close, unit_dir_for

    directory = unit_dir_for(KIND_SESSION, unit_id)
    if directory is None:
        return False
    closed = _newest_close(directory)
    return closed is not None and closed.data.get("reason") == SUPERSEDED_CLOSE_REASON


def _home() -> str:
    from kiro_crew.config.paths import data_home

    return str(data_home())

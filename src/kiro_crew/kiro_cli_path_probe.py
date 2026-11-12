"""The version of a kiro-cli found only through ``PATH``, read inside the sandbox.

A session whose kiro-cli is the pinned install has its version read by
``kiro_cli.kiro_cli_version_at``, outside the sandbox, because the pin never
names a file an agent could have planted. A kiro-cli reached only through
``PATH`` has no such guarantee: ``PATH`` can lead with a directory the agent
writes to. The session itself runs that file inside the OS sandbox, so the rule
here is that the probe never has more reach than that launch:

* It runs through ``sandbox.sandboxed_spawn_argv(mode="strict")``, the chokepoint
  every agent-influenced spawn uses, entered through the absolute system
  ``env`` the readiness probe uses for an unverified candidate, so a file named
  ``kiro-cli`` is never taken for the trusted spawn that may delegate to the
  harness's own sandbox.
* A wrap that did not put the child in Kiro Crew's own sandbox is not run.
* Its environment is the readiness probe's fixed ``--version`` allowlist: no
  credential, no proxy.
* Windows has no OS sandbox backend, so nothing runs there.

Every failure is "unknown", which the caller reads as "keep the full tool list".
The answer is never waited for: :func:`version_if_known` reads a cache and, on a
miss, starts the probe in the background. One answer per file identity (path and
mtime), so a replaced file is read again.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

_Version = tuple[int, int, int]

#: ``None`` is an answer too: the probe ran and established no version.
_versions: dict[tuple[str, int], _Version | None] = {}
_in_flight: set[tuple[str, int]] = set()
_lock = threading.Lock()


def _identity(binary: str) -> tuple[str, int] | None:
    try:
        return (binary, os.stat(binary).st_mtime_ns)
    except OSError:
        return None


def version_if_known(binary: str) -> _Version | None:
    """The cached version of *binary*, or ``None`` while it is unknown.

    Never spawns and never waits. A file with no answer yet has its probe
    started in the background, so a later caller finds one.
    """
    key = _identity(binary)
    if key is None:
        return None
    with _lock:
        if key in _versions:
            return _versions[key]
        if key in _in_flight:
            return None
        _in_flight.add(key)
    try:
        threading.Thread(
            target=_probe_and_record, args=(key,), name="kiro-cli-path-probe", daemon=True
        ).start()
    except RuntimeError:
        with _lock:
            _in_flight.discard(key)
    return None


def warm() -> None:
    """Start the probe for this host's ``PATH``-only install, if it has one.

    Returns at once: finding the install stats directories and the probe spawns,
    so both run on a thread of their own.
    """

    def _warm() -> None:
        try:
            from kiro_crew import kiro_cli

            binary = kiro_cli.path_only_install()
            if binary:
                version_if_known(binary)
        except Exception:  # noqa: BLE001 - a warm-up that cannot run changes nothing
            logger.debug("kiro-cli PATH probe warm-up failed", exc_info=True)

    with contextlib.suppress(RuntimeError):
        threading.Thread(target=_warm, name="kiro-cli-path-probe-warm", daemon=True).start()


def _probe_and_record(key: tuple[str, int]) -> None:
    version: _Version | None = None
    started = time.monotonic()
    try:
        version = _sandboxed_version(key[0])
    except Exception as exc:  # noqa: BLE001 - an unanswerable probe is "unknown"
        # The class only: the message can carry a path read back from the wrap.
        logger.debug("kiro-cli PATH probe failed: %s", type(exc).__name__)
    with _lock:
        _versions[key] = version
        _in_flight.discard(key)
    elapsed = time.monotonic() - started
    if version:
        logger.info(
            "kiro-cli found only through PATH: version %s, read inside the sandbox in %.2fs",
            ".".join(map(str, version)),
            elapsed,
        )
    else:
        logger.info(
            "kiro-cli found only through PATH: no version could be read inside the "
            "sandbox (%.2fs), so Kiro Crew's own MCP servers stay out of Tool Search "
            "deferral for it",
            elapsed,
        )


def _sandboxed_version(binary: str) -> _Version | None:
    """One bounded ``--version`` of *binary* inside the strict sandbox. Blocking."""
    if sys.platform == "win32":
        return None
    from kiro_crew import platform_compat, sandbox
    from kiro_crew.kiro_cli import _VERSION_PROBE_TIMEOUT_SECS
    from kiro_crew.kiro_prerequisite import _probe_env
    from kiro_crew.mcp_hot_reload import parse_kiro_cli_version

    # The system's own ``env``, found in the fixed system directories and never
    # through ``PATH``. Entering through it keeps a file named ``kiro-cli`` from
    # being taken for the trusted spawn that may delegate to the harness's sandbox.
    system_env = platform_compat.trusted_system_bin("env")
    if not system_env:
        return None
    cleanup: str | None = None
    try:
        argv, env, cleanup = sandbox.sandboxed_spawn_argv(
            [system_env, binary, "--version"],
            mode="strict",
            env=_probe_env(dict(os.environ), os.environ.get("PATH", "")),
            strip_python_env=True,
        )
        if not sandbox.wrapped_by_crew_sandbox(argv):
            return None
        if not os.path.isabs(argv[0]):
            # The wrapper is looked up in the system directories only: the child's
            # PATH is the one that may lead with an agent-writable directory. The
            # quiet lookup, because nothing read back from the wrap is logged.
            wrapper = platform_compat.trusted_system_bin_quiet(argv[0])
            if not wrapper:
                return None
            argv[0] = wrapper
        proc = sandbox.popen_limited(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Its own process group: the sandbox launcher forks the file it runs,
            # so a timeout has to take the group, not the launcher alone.
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=_VERSION_PROBE_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            platform_compat.kill_popen_tree(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.communicate(timeout=_VERSION_PROBE_TIMEOUT_SECS)
            return None
        if proc.returncode != 0:
            return None
        text = out.decode("utf-8", "replace") or err.decode("utf-8", "replace")
        return parse_kiro_cli_version(text)
    finally:
        if cleanup:
            with contextlib.suppress(OSError):
                os.unlink(cleanup)

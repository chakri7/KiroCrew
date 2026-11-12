"""A kiro-cli found only through ``PATH`` is versioned inside the sandbox, off the spawn path.

``PATH`` can lead with a directory an agent writes to, so the probe never has
more reach than the session's own launch of the same file: strict sandbox, the
readiness probe's ``--version`` environment, nothing run when the wrap did not
apply, nothing run on Windows. Every failure is "unknown". The caller never
waits: a miss answers ``None`` and starts the probe in the background.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiro_crew import kiro_cli, kiro_cli_path_probe, platform_compat, sandbox

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="Windows has no OS sandbox backend")


def _install(tmp_path: Path, name: str = "kiro-cli") -> str:
    path = tmp_path / "pathdir" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("executable")
    path.chmod(0o755)
    return str(path)


class _Threads:
    """Stand-in for ``threading.Thread``: records the work and runs it when told."""

    def __init__(self) -> None:
        self.pending: list = []

    def __call__(self, *, target, args=(), name="", daemon=False):
        assert daemon is True
        self.pending.append((target, args))
        return self

    def start(self) -> None:
        return None

    def run_all(self) -> None:
        pending, self.pending = self.pending, []
        for target, args in pending:
            target(*args)


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> _Threads:
    fake = _Threads()
    monkeypatch.setattr(kiro_cli_path_probe.threading, "Thread", fake)
    return fake


class _Proc:
    def __init__(self, out: bytes = b"", err: bytes = b"", returncode: int = 0, hang: bool = False):
        self._out, self._err, self.returncode, self._hang = out, err, returncode, hang
        self.killed = False

    def communicate(self, timeout=None):
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired("kiro-cli", timeout)
        return self._out, self._err

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def chokepoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The real ``_sandboxed_version`` over a recorded sandbox chokepoint."""
    calls: dict = {"wrap": [], "popen": [], "proc": _Proc(out=b"kiro-cli 3.2.1\n")}
    launcher = tmp_path / "kirocrew-sandbox-launcher.sh"
    launcher.write_text("#!/bin/sh\n")
    calls["launcher"] = launcher

    def _wrap(argv, mode="standard", *, env=None, **options):
        calls["wrap"].append({"argv": list(argv), "mode": mode, "env": dict(env or {}), **options})
        return ["/bin/sh", str(launcher), *argv], dict(env or {}), str(launcher)

    def _popen(argv, **kwargs):
        calls["popen"].append({"argv": list(argv), **kwargs})
        return calls["proc"]

    def _kill_tree(proc):
        calls["tree_kills"].append(proc)
        proc.kill()

    calls["tree_kills"] = []
    # The system directories of this host are not the subject: ``env`` and any
    # wrapper resolve to one fixed place.
    monkeypatch.setattr(platform_compat, "trusted_system_bin", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        platform_compat, "trusted_system_bin_quiet", lambda name: "/usr/bin/" + name
    )
    monkeypatch.setattr(sandbox, "sandboxed_spawn_argv", _wrap)
    monkeypatch.setattr(sandbox, "wrapped_by_crew_sandbox", lambda argv: True)
    monkeypatch.setattr(sandbox, "popen_limited", _popen)
    monkeypatch.setattr(platform_compat, "kill_popen_tree", _kill_tree)
    monkeypatch.setattr(
        kiro_cli_path_probe, "_sandboxed_version", _REAL_SANDBOXED_VERSION, raising=True
    )
    return calls


# Captured at import, before the conftest fixture stubs it for every test.
_REAL_SANDBOXED_VERSION = kiro_cli_path_probe._sandboxed_version


class TestNeverWaitedFor:
    def test_a_miss_answers_unknown_and_starts_one_probe(self, tmp_path, threads, monkeypatch):
        binary = _install(tmp_path)
        monkeypatch.setattr(kiro_cli_path_probe, "_sandboxed_version", lambda _b: (3, 2, 1))

        assert kiro_cli_path_probe.version_if_known(binary) is None
        assert kiro_cli_path_probe.version_if_known(binary) is None
        assert len(threads.pending) == 1

        threads.run_all()
        assert kiro_cli_path_probe.version_if_known(binary) == (3, 2, 1)
        assert threads.pending == []

    def test_an_unknown_answer_is_kept_and_not_probed_again(self, tmp_path, threads, monkeypatch):
        binary = _install(tmp_path)
        probed: list[str] = []
        monkeypatch.setattr(
            kiro_cli_path_probe, "_sandboxed_version", lambda b: probed.append(b) or None
        )

        kiro_cli_path_probe.version_if_known(binary)
        threads.run_all()

        assert kiro_cli_path_probe.version_if_known(binary) is None
        assert threads.pending == []
        assert probed == [binary]

    def test_a_probe_that_raises_is_unknown(self, tmp_path, threads, monkeypatch):
        binary = _install(tmp_path)

        def _boom(_binary):
            raise OSError("sandbox unavailable")

        monkeypatch.setattr(kiro_cli_path_probe, "_sandboxed_version", _boom)
        kiro_cli_path_probe.version_if_known(binary)
        threads.run_all()

        assert kiro_cli_path_probe.version_if_known(binary) is None
        assert threads.pending == []

    def test_a_replaced_file_is_read_again(self, tmp_path, threads, monkeypatch):
        binary = _install(tmp_path)
        answers = iter([(3, 2, 1), (2, 20, 0)])
        monkeypatch.setattr(kiro_cli_path_probe, "_sandboxed_version", lambda _b: next(answers))
        kiro_cli_path_probe.version_if_known(binary)
        threads.run_all()
        assert kiro_cli_path_probe.version_if_known(binary) == (3, 2, 1)

        stat = os.stat(binary)
        os.utime(binary, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        assert kiro_cli_path_probe.version_if_known(binary) is None
        threads.run_all()
        assert kiro_cli_path_probe.version_if_known(binary) == (2, 20, 0)

    def test_a_missing_file_is_unknown_and_starts_nothing(self, tmp_path, threads):
        assert kiro_cli_path_probe.version_if_known(str(tmp_path / "gone")) is None
        assert threads.pending == []


class TestWarm:
    def test_it_probes_the_path_only_install(self, tmp_path, threads, monkeypatch):
        binary = _install(tmp_path)
        monkeypatch.setattr(kiro_cli, "path_only_install", lambda: binary)
        monkeypatch.setattr(kiro_cli_path_probe, "_sandboxed_version", lambda _b: (3, 0, 0))

        kiro_cli_path_probe.warm()
        threads.run_all()  # the lookup
        threads.run_all()  # the probe it started

        assert kiro_cli_path_probe.version_if_known(binary) == (3, 0, 0)

    @pytest.mark.parametrize("install", [None, "raises"], ids=["no-path-install", "lookup-fails"])
    def test_it_starts_no_probe_without_one(self, install, threads, monkeypatch):
        def _lookup():
            if install == "raises":
                raise OSError("unreadable home")
            return None

        monkeypatch.setattr(kiro_cli, "path_only_install", _lookup)
        kiro_cli_path_probe.warm()
        threads.run_all()

        assert threads.pending == []


@posix_only
class TestTheProbeItself:
    def test_it_runs_one_file_through_the_strict_chokepoint(self, tmp_path, chokepoint):
        binary = _install(tmp_path)

        assert kiro_cli_path_probe._sandboxed_version(binary) == (3, 2, 1)

        (wrap,) = chokepoint["wrap"]
        assert wrap["argv"] == ["/usr/bin/env", binary, "--version"]
        assert wrap["mode"] == "strict"
        assert "is_kiro_cli" not in wrap and "first_party_fixed_argv" not in wrap
        (popen,) = chokepoint["popen"]
        assert popen["argv"][-3:] == ["/usr/bin/env", binary, "--version"]
        assert popen["stdin"] == subprocess.DEVNULL
        assert not chokepoint["launcher"].exists(), "the launcher is removed after the run"

    def test_its_environment_is_the_version_allowlist(self, tmp_path, chokepoint, monkeypatch):
        from kiro_crew.kiro_prerequisite import _PROBE_ENV_KEYS

        for name in (
            "KIRO_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "SLACK_BOT_TOKEN",
            "SSH_AUTH_SOCK",
            "HTTPS_PROXY",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.setenv(name, "secret-value")
        monkeypatch.setenv("HOME", str(tmp_path))

        kiro_cli_path_probe._sandboxed_version(_install(tmp_path))

        env = chokepoint["wrap"][0]["env"]
        assert set(env) <= _PROBE_ENV_KEYS | {"NO_COLOR", "TERM"}
        assert "secret-value" not in env.values()
        assert env["HOME"] == str(tmp_path)
        assert chokepoint["popen"][0]["env"] == env

    def test_a_wrap_outside_the_crew_sandbox_is_not_run(self, tmp_path, chokepoint, monkeypatch):
        monkeypatch.setattr(sandbox, "wrapped_by_crew_sandbox", lambda argv: False)

        assert kiro_cli_path_probe._sandboxed_version(_install(tmp_path)) is None
        assert chokepoint["popen"] == []
        assert not chokepoint["launcher"].exists()

    def test_a_sandbox_that_cannot_be_built_runs_nothing(self, tmp_path, chokepoint, monkeypatch):
        """The refusal reaches ``_probe_and_record``, which records "unknown"."""

        def _refuse(*_a, **_k):
            raise RuntimeError("no sandbox backend")

        monkeypatch.setattr(sandbox, "sandboxed_spawn_argv", _refuse)
        with pytest.raises(RuntimeError):
            kiro_cli_path_probe._sandboxed_version(_install(tmp_path))
        assert chokepoint["popen"] == []

    def test_a_bare_wrapper_name_resolves_in_system_directories_only(
        self, tmp_path, chokepoint, monkeypatch
    ):
        binary = _install(tmp_path)
        monkeypatch.setattr(
            sandbox,
            "sandboxed_spawn_argv",
            lambda argv, mode="standard", *, env=None, **_o: (["bwrap", *argv], dict(env), None),
        )
        monkeypatch.setattr(platform_compat, "trusted_system_bin_quiet", lambda name: None)
        assert kiro_cli_path_probe._sandboxed_version(binary) is None
        assert chokepoint["popen"] == []

        monkeypatch.setattr(
            platform_compat, "trusted_system_bin_quiet", lambda name: f"/usr/bin/{name}"
        )
        assert kiro_cli_path_probe._sandboxed_version(binary) == (3, 2, 1)
        assert chokepoint["popen"][0]["argv"][0] == "/usr/bin/bwrap"

    def test_no_system_env_runs_nothing(self, tmp_path, chokepoint, monkeypatch):
        """``env`` comes from the system directories or the read does not happen."""
        monkeypatch.setattr(platform_compat, "trusted_system_bin", lambda name: None)

        assert kiro_cli_path_probe._sandboxed_version(_install(tmp_path)) is None
        assert chokepoint["wrap"] == [] and chokepoint["popen"] == []

    @pytest.mark.parametrize(
        "outcome",
        [
            {"out": b"kiro-cli 3.2.1\n", "returncode": 1},
            {"out": b"no version here\n"},
            {},
            {"hang": True},
        ],
        ids=["non-zero-exit", "unparsable", "silent", "timeout"],
    )
    def test_every_failed_run_is_unknown(self, outcome, tmp_path, chokepoint):
        proc = chokepoint["proc"] = _Proc(**outcome)

        assert kiro_cli_path_probe._sandboxed_version(_install(tmp_path)) is None
        assert not chokepoint["launcher"].exists()
        assert chokepoint["tree_kills"] == ([proc] if outcome.get("hang") else [])

    def test_the_child_gets_its_own_process_group(self, tmp_path, chokepoint):
        """The sandbox launcher forks the file it runs, so a timeout must take the group."""
        kiro_cli_path_probe._sandboxed_version(_install(tmp_path))

        assert chokepoint["popen"][0]["start_new_session"] is True


def test_windows_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(kiro_cli_path_probe.sys, "platform", "win32")

    def _never(*_a, **_k):
        pytest.fail("Windows has no OS sandbox backend: the probe must not run")

    monkeypatch.setattr(sandbox, "sandboxed_spawn_argv", _never)
    monkeypatch.setattr(sandbox, "popen_limited", _never)

    assert _REAL_SANDBOXED_VERSION(_install(tmp_path)) is None

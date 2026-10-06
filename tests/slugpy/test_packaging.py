"""
Tests for the pieces of slugpy that support a pip-installed copy of
slug: data search paths (slugpy._paths), the slug console-script
launcher (slugpy._cli), and the data downloader's defaults
(slugpy.download_data).

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
import sys
from pathlib import Path

import pytest

from slugpy import _cli, _paths, download_data


@pytest.fixture
def fake_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """
    Point user_data_dir and package_share_dir at scratch directories.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test scratch directory.
    monkeypatch : pytest.MonkeyPatch
        pytest's monkeypatch fixture.

    Returns
    -------
    tuple of pathlib.Path
        The (user data, package share) directories; the share directory
        exists, the user data directory does not.
    """
    user = tmp_path / "user"
    share = tmp_path / "share"
    share.mkdir()
    monkeypatch.setattr(_paths, "user_data_dir", lambda: user)
    monkeypatch.setattr(_paths, "package_share_dir", lambda: share)
    monkeypatch.delenv(_paths.SLUG_DATA_PATH_VAR, raising=False)
    return user, share


def test_ensure_search_path_order_and_idempotence(
        fake_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    user, share = fake_dirs
    monkeypatch.setenv(_paths.SLUG_DATA_PATH_VAR, "/already/there")
    _paths.ensure_search_path()
    expected = os.pathsep.join([str(user), str(share), "/already/there"])
    assert os.environ[_paths.SLUG_DATA_PATH_VAR] == expected
    _paths.ensure_search_path()
    assert os.environ[_paths.SLUG_DATA_PATH_VAR] == expected


def test_ensure_search_path_skips_missing_share(
        fake_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user, _ = fake_dirs
    monkeypatch.setattr(_paths, "package_share_dir", lambda: tmp_path / "absent")
    _paths.ensure_search_path()
    assert os.environ[_paths.SLUG_DATA_PATH_VAR] == str(user)


def test_default_data_root_honors_slug_dir(
        fake_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user, _ = fake_dirs
    monkeypatch.setenv("SLUG_DIR", str(tmp_path / "slugdir"))
    assert _paths.default_data_root() == tmp_path / "slugdir"
    assert download_data._default_dest() == tmp_path / "slugdir" / "data"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("SLUG_DIR", "")
    assert _paths.default_data_root() == user
    monkeypatch.delenv("SLUG_DIR")
    assert download_data._default_dest() == user / "data"  # pyright: ignore[reportPrivateUsage]


def test_cli_missing_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_cli, "slug_executable", lambda: tmp_path / "no_such_slug")
    with pytest.raises(SystemExit, match="not found"):
        _cli.main()


def test_cli_execs_bundled_executable(
        fake_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user, share = fake_dirs
    exe = tmp_path / "slug"
    exe.write_text("")
    calls: list[tuple[Path, list[str]]] = []

    def fake_execv(path: Path, args: list[str]) -> None:
        calls.append((path, args))

    monkeypatch.setattr(_cli, "slug_executable", lambda: exe)
    monkeypatch.setattr(os, "execv", fake_execv)
    monkeypatch.setattr(sys, "argv", ["slug", "deck.in", "--restart"])
    _cli.main()
    assert calls == [(exe, [str(exe), "deck.in", "--restart"])]
    assert os.environ[_paths.SLUG_DATA_PATH_VAR].split(os.pathsep)[:2] == [str(user), str(share)]


def test_deep_merge_toml_keeps_local_entries() -> None:
    local: dict[str, object] = {"a": {"x": 1, "local_only": 2}, "lst": [1, 2], "s": "old"}
    remote: dict[str, object] = {"a": {"x": 10}, "lst": [3, 1], "s": "new"}
    download_data.deep_merge_toml(local, remote)  # pyright: ignore[reportArgumentType]
    assert local == {"a": {"x": 10, "local_only": 2}, "lst": [3, 1, 2], "s": "new"}


def test_install_toml_writes_and_merges_without_leftovers(tmp_path: Path) -> None:
    dest = tmp_path / "reg.toml"
    download_data.install_toml(b'a = 1\n[t]\nx = 1\n', dest, overwrite=False, verbose=False)
    assert dest.read_text() == 'a = 1\n[t]\nx = 1\n'
    dest.write_text('a = 1\nlocal = 5\n[t]\nx = 1\n')
    download_data.install_toml(b'a = 2\n[t]\nx = 3\n', dest, overwrite=False, verbose=False)
    merged = dest.read_text()
    assert "a = 2" in merged and "local = 5" in merged and "x = 3" in merged
    assert sorted(f.name for f in tmp_path.iterdir()) == ["reg.toml"]

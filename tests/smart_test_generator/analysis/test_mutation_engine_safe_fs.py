from pathlib import Path
import pytest

from smart_test_generator.analysis.mutation_engine import SafeFS


def test_safe_fs_blocks_escape(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    safe = SafeFS(sandbox)

    outside = tmp_path / "outside.txt"
    with pytest.raises(ValueError):
        safe.write_text(outside, "nope")


def test_safe_fs_allows_within(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    safe = SafeFS(sandbox)

    inside = sandbox / "dir" / "file.txt"
    safe.write_text(inside, "ok")
    assert inside.read_text() == "ok"


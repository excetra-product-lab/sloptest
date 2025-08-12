from pathlib import Path
from smart_test_generator.utils.writer import TestFileWriter


def test_write_or_merge_ast_merge_adds_content(tmp_path: Path):
    writer = TestFileWriter(str(tmp_path))
    src = tmp_path / "src" / "sample.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def add(a,b):\n    return a+b\n")

    # initial write creates tests file
    new_test = """
import os
def test_add():
    assert True
"""
    res = writer.write_or_merge_test_file(str(src), new_test, strategy='ast-merge')
    assert res.changed is True

    test_path = tmp_path / "tests" / "test_sample.py"
    assert test_path.exists()

    # merging new function should update without duplicating
    updated = """
import os
def test_add():
    assert True
def test_add_more():
    assert True
"""
    res2 = writer.write_or_merge_test_file(str(src), updated, strategy='ast-merge')
    assert res2.changed is True
    content = test_path.read_text()
    assert content.count("def test_add(") == 1
    assert "def test_add_more(" in content


def test_write_or_merge_idempotent_noop(tmp_path: Path):
    writer = TestFileWriter(str(tmp_path))
    src = tmp_path / "src" / "math.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def mul(a,b):\n    return a*b\n")

    body = """
import math
def test_mul():
    assert 6 == 2*3
"""
    res1 = writer.write_or_merge_test_file(str(src), body, strategy='ast-merge')
    assert res1.changed is True

    # Second run with identical content should be a no-op
    res2 = writer.write_or_merge_test_file(str(src), body, strategy='ast-merge')
    assert res2.changed is False


def test_write_or_merge_dry_run_diff(tmp_path: Path):
    writer = TestFileWriter(str(tmp_path))
    src = tmp_path / "pkg" / "util.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def foo():\n    return 1\n")

    content = """
def test_foo():
    assert True
"""
    res = writer.write_or_merge_test_file(str(src), content, strategy='ast-merge', dry_run=True)
    assert res.changed is True
    assert isinstance(res.diff, str)
    assert '\n' in res.diff

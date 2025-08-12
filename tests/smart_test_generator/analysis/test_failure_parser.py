from pathlib import Path
from smart_test_generator.analysis.coverage.failure_parser import (
    parse_stdout_stderr,
    write_failures_json,
    parse_junit_xml,
)


def test_parse_stdout_stderr_extracts_basic_failure(tmp_path: Path):
    stdout = "tests/test_sample.py::test_x FAILED\n"
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.nodeid.startswith("tests/test_sample.py::test_x")
    out = write_failures_json(parsed, tmp_path)
    assert out.exists()


def test_parse_junit_xml_handles_empty(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuite></testsuite>")
    parsed = parse_junit_xml(junit)
    assert parsed.total == 0

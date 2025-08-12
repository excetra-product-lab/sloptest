from pathlib import Path
from smart_test_generator.generation.refine.payload_builder import build_payload, write_payload_json, build_refine_prompt
from smart_test_generator.analysis.coverage.failure_parser import ParsedFailures, FailureRecord
from smart_test_generator.config import Config


def test_build_payload_and_prompt(tmp_path: Path):
    cfg = Config()
    failures = ParsedFailures(
        total=1,
        failures=[
            FailureRecord(
                nodeid="tests/test_a.py::test_x",
                file="tests/test_a.py",
                line=3,
                message="assert 1 == 2",
                assertion_diff="assert 1 == 2",
                captured_stdout=None,
                captured_stderr=None,
                duration=0.01,
            )
        ],
    )
    payload = build_payload(
        failures=failures,
        project_root=tmp_path,
        config=cfg,
        tests_written=["tests/test_a.py"],
        last_run_command=["python", "-m", "pytest"],
    )
    out = write_payload_json(payload, tmp_path / ".artifacts" / "refine" / "test")
    assert out.exists()
    prompt = build_refine_prompt(payload, cfg)
    assert "Refine failing tests" in prompt
    assert "Failures (top):" in prompt

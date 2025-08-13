import json
from pathlib import Path
from unittest.mock import patch

from smart_test_generator.config import Config
from smart_test_generator.services.test_generation_service import TestGenerationService
from smart_test_generator.generation.llm_clients import LLMClient
from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage
from smart_test_generator.utils.user_feedback import UserFeedback


class DummyLLM(LLMClient):
    def __init__(self):
        pass

    def generate_unit_tests(self, system_prompt, xml_content, directory_structure, source_files, project_root):
        # Return a simple failing test for the given source file
        out = {}
        for src in source_files:
            test_path = Path(src)
            name = f"test_{test_path.stem}.py"
            out[str(test_path)] = (
                "import pytest\n\n"
                "def test_fails():\n"
                "    assert 1 == 2\n"
            )
        return out

    def refine_tests(self, request):
        # Return updated files that fix the failure
        return json.dumps({
            "updated_files": [{"path": "tests/test_sample.py", "content": "def test_fails():\n    assert 1 == 1\n"}],
            "rationale": "Fix failing assertion",
            "plan": "Update test"
        })


def _make_plan(tmp_path: Path) -> TestGenerationPlan:
    # Create a dummy source file
    src = tmp_path / "sample.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    cov = TestCoverage(
        filepath=str(src),
        line_coverage=0.0,
        branch_coverage=0.0,
        missing_lines=[1, 2],
        covered_functions=set(),
        uncovered_functions={"add"},
    )
    return TestGenerationPlan(
        source_file=str(src),
        elements_to_test=[type("E", (), {"name": "add"})()],
        existing_test_files=[],
        coverage_before=cov,
        estimated_coverage_after=50.0,
    )


def test_generate_run_refine_flow(tmp_path: Path):
    cfg = Config()
    # Enable auto-run and refine
    cfg.config['test_generation']['generation']['test_runner'] = {
        'enable': True,
        'args': ['-q'],
    }
    cfg.config['test_generation']['generation']['refine'] = {
        'enable': True,
        'max_retries': 2,
        'backoff_base_sec': 0.01,
        'backoff_max_sec': 0.02,
        'stop_on_no_change': True,
    }

    feedback = UserFeedback(quiet=True)
    svc = TestGenerationService(project_root=tmp_path, config=cfg, feedback=feedback)

    plan = _make_plan(tmp_path)

    # Patch run_pytest to fail once then pass
    call_count = {'n': 0}

    def fake_run_pytest(spec, junit_xml=False):
        class R:
            def __init__(self, returncode, cwd, cmd):
                self.returncode = returncode
                self.cwd = str(spec.cwd)
                self.stdout = "sample.py::test_fails FAILED\nassert 1 == 2\n"
                self.stderr = ""
                self.cmd = list(spec.argv)
                self.junit_xml_path = None
        call_count['n'] += 1
        # First call fails, second call passes
        if call_count['n'] == 1:
            # ensure artifacts dir exists to simulate runner behavior
            (spec.cwd / ".artifacts" / "coverage" / "0001").mkdir(parents=True, exist_ok=True)
            return R(1, spec.cwd, spec.argv)
        return R(0, spec.cwd, spec.argv)

    with patch('smart_test_generator.analysis.coverage.runner.run_pytest', side_effect=fake_run_pytest):
        result = svc.generate_tests(
            llm_client=DummyLLM(),
            test_plans=[plan],
            directory_structure="",
            batch_size=1,
            generation_reasons={str(Path(plan.source_file)): "Generated"},
        )

    # Test file should be written
    test_file = tmp_path / "tests" / "test_sample.py"
    assert test_file.exists()

    # After refinement, the updated content should make the test pass on the second run
    content = test_file.read_text()
    assert "assert 1 == 1" in content

    # Artifacts for refine should exist
    refine_root = tmp_path / ".artifacts" / "refine"
    assert refine_root.exists()


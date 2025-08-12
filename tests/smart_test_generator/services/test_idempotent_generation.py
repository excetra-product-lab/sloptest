from pathlib import Path
from unittest.mock import Mock

import pytest

from smart_test_generator.config import Config
from smart_test_generator.services.test_generation_service import TestGenerationService


def _make_source(tmp_path: Path, name: str = "sample.py") -> Path:
    src = tmp_path / "src" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def add(a, b):\n    return a + b\n")
    return src


def _make_plan(src: Path):
    # Minimal TestGenerationPlan-like object using real dataclass import is heavier here
    from smart_test_generator.models.data_models import TestGenerationPlan
    return TestGenerationPlan(
        source_file=str(src),
        elements_to_test=[],
        existing_test_files=[],
        coverage_before=None,
        estimated_coverage_after=50.0,
    )


def test_service_generation_is_idempotent_with_ast_merge(tmp_path: Path):
    # Config with AST merge enabled
    cfg = Config()
    cfg.config['test_generation']['generation']['merge'] = {
        'strategy': 'ast-merge',
        'dry_run': False,
        'formatter': 'none',
    }

    # Minimal feedback mock used by the service
    feedback = Mock()
    feedback.console = Mock()
    feedback.console.print = Mock()
    feedback.sophisticated_progress = Mock()
    feedback.status_spinner = Mock()
    feedback.status_spinner.return_value.__enter__ = Mock()
    feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
    feedback.success = Mock()
    feedback.warning = Mock()
    feedback.error = Mock()

    svc = TestGenerationService(project_root=tmp_path, config=cfg, feedback=feedback)

    src = _make_source(tmp_path)
    plan = _make_plan(src)

    # Build a deterministic test body that should be merged idempotently
    test_body = (
        "import os\n\n"
        "def test_add():\n"
        "    assert 5 == (2+3)\n"
    )

    # Patch the incremental client on the service to return our deterministic output
    class FakeIncremental:
        def __init__(self, *args, **kwargs):
            pass

        def generate_contextual_tests(self, plans, directory_structure, batch_source_files, project_root):
            return {str(src): test_body}

    # Inject fake client
    from smart_test_generator.services import test_generation_service as tgs_mod
    tgs_mod.IncrementalLLMClient = FakeIncremental  # type: ignore

    # First run writes file
    result1 = svc.generate_tests(llm_client=Mock(), test_plans=[plan], directory_structure="", batch_size=1)
    assert str(src) in result1

    test_file = tmp_path / "tests" / f"test_{src.stem}.py"
    assert test_file.exists()
    content1 = test_file.read_text()
    assert content1.count("def test_add(") == 1

    # Second run should be a no-op (no duplicate functions)
    result2 = svc.generate_tests(llm_client=Mock(), test_plans=[plan], directory_structure="", batch_size=1)
    assert str(src) in result2
    content2 = test_file.read_text()
    assert content2.count("def test_add(") == 1
    assert content2 == content1


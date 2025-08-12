import json
from pathlib import Path
from unittest.mock import Mock

from smart_test_generator.generation.refine.refine_manager import run_refinement_cycle
from smart_test_generator.config import Config


def test_run_refinement_cycle_basic(tmp_path: Path):
    cfg = Config()
    cfg.config['test_generation']['generation']['refine'] = {
        'enable': True,
        'max_retries': 2,
        'backoff_base_sec': 0.01,
        'backoff_max_sec': 0.02,
        'stop_on_no_change': True,
    }

    payload = {
        'run_id': '123',
        'failures': [{'nodeid': 'tests/test_x.py::test_y', 'file': 'tests/test_x.py', 'message': 'fail'}],
        'config_summary': {'style': {'framework': 'pytest'}},
    }

    class DummyLLM:
        def refine_tests(self, request):
            # Return one updated file then stop
            return json.dumps({
                'updated_files': [{'path': 'tests/test_x.py', 'content': 'def test_y():\n    assert True\n'}],
                'rationale': 'Fix', 'plan': 'Update test'
            })

    applied = {}

    def apply_updates(files, project_root: Path):
        for f in files:
            p = project_root / f['path']
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f['content'])
            applied[p] = True

    reruns = [1, 0]

    def re_run():
        return reruns.pop(0)

    out = run_refinement_cycle(
        payload=payload,
        project_root=tmp_path,
        artifacts_dir=tmp_path / '.artifacts' / 'refine' / '123',
        llm_client=DummyLLM(),
        config=cfg,
        apply_updates_fn=apply_updates,
        re_run_pytest_fn=re_run,
    )

    assert out.updated_any is True
    assert out.final_exit_code == 0

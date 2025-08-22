#!/usr/bin/env python3
"""Test script to verify the refinement flow is working correctly."""

from smart_test_generator.generation.refine.refine_manager import run_refinement_cycle
from smart_test_generator.config import Config
from pathlib import Path
import tempfile
import json


def test_refinement_flow():
    """Test the complete refinement flow end-to-end."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test config with refinement enabled
        cfg = Config()
        cfg.config = {
            'test_generation': {
                'generation': {
                    'refine': {
                        'enable': True,
                        'max_retries': 2,
                        'backoff_base_sec': 0.01,
                        'backoff_max_sec': 0.02,
                        'stop_on_no_change': True
                    }
                }
            }
        }

        # Create a simple test LLM client that returns refinement results
        class TestLLMClient:
            def refine_tests(self, request):
                payload = request.get('payload', {})
                prompt = request.get('prompt', '')

                if not prompt or not payload:
                    return json.dumps({
                        'updated_files': [],
                        'rationale': 'No refinement data provided',
                        'plan': 'Cannot proceed without proper refinement context'
                    })

                # Return a successful refinement
                return json.dumps({
                    'updated_files': [{
                        'path': 'test_refined.py',
                        'content': 'def test_refined():\n    assert True  # Refined test\n'
                    }],
                    'rationale': 'Fixed failing test assertion',
                    'plan': 'Updated test with correct assertion'
                })

        # Create payload with test failures
        payload = {
            'run_id': 'test_123',
            'failures': [{
                'nodeid': 'tests/test_example.py::test_failing',
                'file': 'tests/test_example.py',
                'line': 10,
                'message': 'AssertionError: Expected True but got False',
                'assertion_diff': 'Expected: True\\nActual: False'
            }],
            'config_summary': {'style': {'framework': 'pytest'}},
            'tests_written': ['tests/test_example.py'],
            'last_run_command': ['pytest', 'tests/'],
            'failures_total': 1
        }

        # Create artifacts directory
        artifacts_dir = tmp_path / '.artifacts' / 'refine' / 'test_123'

        # Track applied updates
        applied_updates = []

        def apply_updates_fn(files, project_root):
            for f in files:
                applied_updates.append(f)
                print(f'Applied update to {f["path"]}')

        # Mock pytest runner that initially fails, then passes
        run_count = [0]  # Use list to make it mutable in nested function

        def mock_pytest_runner():
            run_count[0] += 1
            print(f'Mock pytest run #{run_count[0]}')
            if run_count[0] <= 1:  # First run fails, second run passes
                print('Mock pytest run: FAILED (exit code 1)')
                return 1  # First run fails
            else:
                print('Mock pytest run: PASSED (exit code 0)')
                return 0  # Subsequent runs pass

        # Run refinement cycle
        print('Starting refinement cycle test...')
        result = run_refinement_cycle(
            payload=payload,
            project_root=tmp_path,
            artifacts_dir=artifacts_dir,
            llm_client=TestLLMClient(),
            config=cfg,
            apply_updates_fn=apply_updates_fn,
            re_run_pytest_fn=mock_pytest_runner
        )

        print("\nRefinement completed:")
        print(f"- Iterations: {result.iterations}")
        print(f"- Final exit code: {result.final_exit_code}")
        print(f"- Updates applied: {result.updated_any}")
        print(f"- Updates count: {len(applied_updates)}")

        if applied_updates:
            print("\nApplied updates:")
            for update in applied_updates:
                print(f"  - {update['path']}")

        # Verify results
        assert result.iterations == 2, f"Expected 2 iterations, got {result.iterations}"
        assert result.final_exit_code == 0, f"Expected final exit code 0, got {result.final_exit_code}"
        assert result.updated_any == True, "Expected updates to be applied"
        assert len(applied_updates) == 2, f"Expected 2 updates, got {len(applied_updates)}"

        print("\n✅ Refinement flow test PASSED!")
        print("✅ The refinement system is working correctly!")

        return True


if __name__ == "__main__":
    test_refinement_flow()

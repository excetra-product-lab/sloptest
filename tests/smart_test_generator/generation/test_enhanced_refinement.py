"""Tests for enhanced refinement system integration."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from smart_test_generator.generation.refine.payload_builder import build_payload, _gather_repo_meta
from smart_test_generator.generation.refine.refine_manager import (
    run_refinement_cycle,
    _determine_retry_strategy,
    _jitter_delay,
    _calculate_average_confidence,
    RefinementOutcome
)
from smart_test_generator.analysis.coverage.failure_parser import FailureRecord, ParsedFailures
from smart_test_generator.analysis.failure_pattern_analyzer import FailureCategory
from smart_test_generator.config import Config
from smart_test_generator.utils.prompt_loader import PromptLoader


@pytest.fixture
def temp_project_root():
    """Create temporary project root directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    config_data = {
        "test_generation": {
            "generation": {
                "refine": {
                    "enable": True,
                    "max_retries": 2,
                    "backoff_base_sec": 1.0,
                    "backoff_max_sec": 8.0,
                    "stop_on_no_change": True,
                    "git_context": {
                        "enable": True,
                        "days_back": 7,
                        "max_commits": 5
                    }
                }
            },
            "style": {
                "framework": "pytest"
            }
        },
        "prompt_engineering": {
            "encourage_step_by_step": True,
            "use_positive_negative_examples": True,
            "decisive_recommendations": True
        }
    }
    # Create a mock config instead of passing dict to Config constructor
    mock_config = Mock(spec=Config)
    
    def get_config_value(key, default=None):
        keys = key.split('.')
        value = config_data
        for k in keys:
            value = value.get(k, {})
            if not isinstance(value, dict):
                break
        return value if value != {} else default
    
    mock_config.get.side_effect = get_config_value
    return mock_config


@pytest.fixture
def sample_failures():
    """Create sample failure records."""
    return ParsedFailures(
        total=4,
        failures=[
            FailureRecord(
                nodeid="test_module.py::test_assert_failure",
                file="test_module.py",
                line=10,
                message="AssertionError: Expected 5, got 3",
                assertion_diff="assert 5 == 3",
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            ),
            FailureRecord(
                nodeid="test_imports.py::test_import_error",
                file="test_imports.py",
                line=5,
                message="ImportError: No module named 'missing_module'",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.05
            ),
            FailureRecord(
                nodeid="test_fixtures.py::test_fixture_error",
                file="test_fixtures.py",
                line=15,
                message="fixture 'db' not found",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.02
            ),
            FailureRecord(
                nodeid="test_mocks.py::test_mock_error",
                file="test_mocks.py",
                line=20,
                message="mock.assert_called_once() failed",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.03
            )
        ]
    )


@pytest.fixture
def mock_git_repo_meta():
    """Mock git repository metadata."""
    return {
        "branch": "feature/test-improvements",
        "commit": "abc123def456",
        "has_uncommitted_changes": False,
        "recent_changes": {
            "time_range": "last 7 days",
            "total_files_changed": 5,
            "test_files_changed": 3,
            "source_files_changed": 2,
            "recent_commit_messages": ["Fix test failures", "Add new tests", "Update logic"],
            "changed_test_files": ["test_module.py", "test_imports.py", "test_new.py"],
            "changed_source_files": ["module.py", "imports.py"]
        },
        "test_changes_detail": [
            {"file": "test_module.py", "status": "M", "lines_added": 5, "lines_removed": 2},
            {"file": "test_imports.py", "status": "A", "lines_added": 20, "lines_removed": 0}
        ]
    }


class TestEnhancedPayloadBuilder:
    """Test enhanced payload building with git and pattern context."""
    
    @patch('smart_test_generator.generation.refine.payload_builder.GitService')
    @patch('smart_test_generator.generation.refine.payload_builder.FailurePatternAnalyzer')
    def test_build_payload_with_git_context(self, mock_analyzer_class, mock_git_service_class, 
                                          temp_project_root, sample_config, sample_failures):
        """Test payload building with git context integration."""
        # Setup mocks
        mock_git_service = Mock()
        mock_git_service_class.return_value = mock_git_service
        
        mock_git_context = Mock()
        mock_git_context.current_branch = "main"
        mock_git_context.current_commit = "abc123"
        mock_git_context.is_dirty = False
        mock_git_context.recent_changes = Mock()
        mock_git_context.recent_changes.time_range = "last 7 days"
        mock_git_context.recent_changes.total_changes = 3
        
        mock_git_service.get_git_context.return_value = mock_git_context
        mock_git_service.get_commit_context_for_failures.return_value = {
            "recent_commits": ["abc123 Fix tests"],
            "changed_test_files": ["test_module.py"]
        }
        mock_git_service.get_recent_test_changes.return_value = []
        
        # Build payload
        payload = build_payload(
            failures=sample_failures,
            project_root=temp_project_root,
            config=sample_config,
            tests_written=["test_module.py", "test_imports.py"],
            last_run_command=["python", "-m", "pytest"],
            include_git_context=True
        )
        
        # Verify git context is included
        assert "repo_meta" in payload
        assert "failure_git_context" in payload
        assert payload["failure_git_context"]["recent_commits"] == ["abc123 Fix tests"]
    
    @patch('smart_test_generator.generation.refine.payload_builder.FailurePatternAnalyzer')
    def test_build_payload_with_pattern_analysis(self, mock_analyzer_class, 
                                                temp_project_root, sample_config, sample_failures):
        """Test payload building with failure pattern analysis."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_analysis = Mock()
        mock_analysis.total_failures = 4
        mock_analysis.pattern_frequencies = {
            FailureCategory.ASSERTION_ERROR: 1,
            FailureCategory.IMPORT_ERROR: 1,
            FailureCategory.FIXTURE_ERROR: 1,
            FailureCategory.MOCK_ERROR: 1
        }
        mock_analysis.trending_patterns = [(FailureCategory.ASSERTION_ERROR, 1.2), (FailureCategory.IMPORT_ERROR, 0.8)]
        mock_analysis.fix_suggestions = [
            Mock(category=FailureCategory.ASSERTION_ERROR, title="Fix assertions", description="Check logic", 
                priority=1, automated=False, code_example="assert x == y")
        ]
        mock_analysis.confidence_scores = {"test1": 0.8, "test2": 0.9}
        
        mock_analyzer.analyze_failures.return_value = mock_analysis
        mock_analyzer.get_success_rates.return_value = {FailureCategory.ASSERTION_ERROR: 0.75}
        
        # Build payload
        payload = build_payload(
            failures=sample_failures,
            project_root=temp_project_root,
            config=sample_config,
            tests_written=["test_module.py"],
            last_run_command=["python", "-m", "pytest"],
            include_pattern_analysis=True
        )
        
        # Verify pattern analysis is included
        assert "failure_analysis" in payload
        analysis = payload["failure_analysis"]
        assert analysis["total_failures"] == 4
        assert "pattern_frequencies" in analysis
        assert "trending_patterns" in analysis
        assert "fix_suggestions" in analysis
        assert "confidence_scores" in analysis
        assert "historical_success_rates" in analysis
    
    def test_build_payload_error_handling(self, temp_project_root, sample_config, sample_failures):
        """Test payload building gracefully handles errors."""
        # Should not crash even with failing git/pattern services
        with patch('smart_test_generator.generation.refine.payload_builder.GitService', 
                   side_effect=Exception("Git error")):
            payload = build_payload(
                failures=sample_failures,
                project_root=temp_project_root,
                config=sample_config,
                tests_written=["test_module.py"],
                last_run_command=["python", "-m", "pytest"],
                include_git_context=True
            )
            
            # Basic payload should still be created
            assert "run_id" in payload
            assert "failures" in payload
            assert payload["failures_total"] == 4


class TestSmartRetryStrategy:
    """Test smart retry strategy determination."""
    
    def test_determine_retry_strategy_assertion_focused(self):
        """Test retry strategy for assertion-heavy failures."""
        payload = {
            "failure_analysis": {
                "total_failures": 5,
                "pattern_frequencies": {
                    "assertion_error": 4,
                    "import_error": 1
                }
            }
        }
        
        strategy, config = _determine_retry_strategy(payload)
        assert strategy == "logic_focused"
        assert config["dominant_category"] == "assertion_error"
        assert config["dominance_ratio"] == 0.8
    
    def test_determine_retry_strategy_dependency_focused(self):
        """Test retry strategy for dependency-related failures."""
        payload = {
            "failure_analysis": {
                "total_failures": 4,
                "pattern_frequencies": {
                    "import_error": 3,
                    "dependency_error": 1
                }
            }
        }

        strategy, config = _determine_retry_strategy(payload)
        assert strategy == "dependency_focused"
        assert config["dominant_category"] == "import_error"
    
    def test_determine_retry_strategy_comprehensive(self):
        """Test retry strategy for diverse failure types."""
        payload = {
            "failure_analysis": {
                "total_failures": 10,
                "pattern_frequencies": {
                    "assertion_error": 2,
                    "import_error": 2,
                    "fixture_error": 2,
                    "mock_error": 2,
                    "type_error": 1,
                    "value_error": 1
                }
            }
        }
        
        strategy, config = _determine_retry_strategy(payload)
        assert strategy == "comprehensive"
    
    def test_determine_retry_strategy_no_analysis(self):
        """Test retry strategy when no failure analysis is available."""
        payload = {}
        
        strategy, config = _determine_retry_strategy(payload)
        assert strategy == "default"
        assert config == {}
    
    def test_jitter_delay_category_adjustments(self):
        """Test delay adjustments based on failure category."""
        base_delay = 2.0
        
        # Import errors should have longer delay
        import_delay = _jitter_delay(base_delay, 8.0, 1, "import_error")
        assert import_delay > base_delay  # Should be increased
        
        # Assertion errors should have shorter delay
        assert_delay = _jitter_delay(base_delay, 8.0, 1, "assertion_error")
        assert assert_delay < base_delay  # Should be decreased
        
        # Unknown category should use base delay
        unknown_delay = _jitter_delay(base_delay, 8.0, 1, "unknown")
        # Allow for jitter variation
        assert 0.8 * base_delay <= unknown_delay <= 1.2 * base_delay
    
    def test_calculate_average_confidence(self):
        """Test average confidence calculation."""
        payload = {
            "failure_analysis": {
                "confidence_scores": {
                    "test1": 0.8,
                    "test2": 0.6,
                    "test3": 0.9
                }
            }
        }
        
        avg = _calculate_average_confidence(payload)
        assert avg == (0.8 + 0.6 + 0.9) / 3
        
        # Test with no confidence scores
        empty_payload = {}
        avg_empty = _calculate_average_confidence(empty_payload)
        assert avg_empty == 0.5  # Default neutral confidence


class TestEnhancedRefinementCycle:
    """Test enhanced refinement cycle with pattern-based logic."""
    
    @patch('smart_test_generator.generation.refine.refine_manager.FailurePatternAnalyzer')
    def test_refinement_cycle_strategy_selection(self, mock_analyzer_class, 
                                                temp_project_root, sample_config):
        """Test refinement cycle uses appropriate strategy."""
        # Setup mocks
        mock_llm_client = Mock()
        mock_llm_client.refine_tests.return_value = '{"updated_files": [], "rationale": "No changes needed"}'
        
        mock_apply_updates = Mock()
        mock_re_run_pytest = Mock(return_value=0)  # Success
        
        payload_with_analysis = {
            "failure_analysis": {
                "pattern_frequencies": {"assertion_error": 3, "import_error": 1},
                "total_failures": 4
            }
        }
        
        artifacts_dir = temp_project_root / "artifacts"
        artifacts_dir.mkdir()
        
        # Run refinement cycle
        outcome = run_refinement_cycle(
            payload=payload_with_analysis,
            project_root=temp_project_root,
            artifacts_dir=artifacts_dir,
            llm_client=mock_llm_client,
            config=sample_config,
            apply_updates_fn=mock_apply_updates,
            re_run_pytest_fn=mock_re_run_pytest
        )
        
        # Verify strategy was applied
        assert isinstance(outcome, RefinementOutcome)
        assert outcome.retry_strategy_used == "logic_focused"
        assert outcome.pattern_insights is not None
        assert "strategy_used" in outcome.pattern_insights
    
    @patch('smart_test_generator.generation.refine.refine_manager.FailurePatternAnalyzer')
    def test_refinement_cycle_max_retries_adjustment(self, mock_analyzer_class,
                                                   temp_project_root, sample_config):
        """Test max retries adjustment based on strategy."""
        mock_llm_client = Mock()
        mock_llm_client.refine_tests.return_value = '{"updated_files": [{"path": "test.py", "content": "fixed"}]}'
        
        mock_apply_updates = Mock()
        mock_re_run_pytest = Mock(return_value=1)  # Keep failing
        
        # Dependency-focused strategy should increase max_retries
        payload = {
            "failure_analysis": {
                "pattern_frequencies": {"dependency_error": 4, "import_error": 1},
                "total_failures": 5
            }
        }
        
        artifacts_dir = temp_project_root / "artifacts"
        artifacts_dir.mkdir()
        
        outcome = run_refinement_cycle(
            payload=payload,
            project_root=temp_project_root,
            artifacts_dir=artifacts_dir,
            llm_client=mock_llm_client,
            config=sample_config,
            apply_updates_fn=mock_apply_updates,
            re_run_pytest_fn=mock_re_run_pytest
        )
        
        # Should use dependency_focused strategy and attempt more retries
        assert outcome.retry_strategy_used == "dependency_focused"
        assert outcome.iterations >= 2  # Should try at least the configured max_retries
    
    def test_refinement_cycle_disabled(self, temp_project_root):
        """Test refinement cycle when disabled in config."""
        disabled_config = Mock(spec=Config)
        disabled_config.get.return_value = False  # refine.enable is False
        
        outcome = run_refinement_cycle(
            payload={},
            project_root=temp_project_root,
            artifacts_dir=temp_project_root / "artifacts",
            llm_client=Mock(),
            config=disabled_config,
            apply_updates_fn=Mock(),
            re_run_pytest_fn=Mock()
        )
        
        assert outcome.iterations == 0
        assert outcome.final_exit_code == 1
        assert not outcome.updated_any
        assert outcome.retry_strategy_used == "disabled"
    
    @patch('smart_test_generator.generation.refine.refine_manager.FailurePatternAnalyzer')
    def test_refinement_success_tracking(self, mock_analyzer_class, 
                                       temp_project_root, sample_config):
        """Test that successful refinements are tracked for learning."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_llm_client = Mock()
        mock_llm_client.refine_tests.return_value = '{"updated_files": [{"path": "test.py", "content": "fixed"}]}'
        
        mock_apply_updates = Mock()
        mock_re_run_pytest = Mock(return_value=0)  # Success after refinement
        
        payload = {
            "failure_analysis": {
                "pattern_frequencies": {"assertion_error": 2, "fixture_error": 1},
                "total_failures": 3
            }
        }
        
        artifacts_dir = temp_project_root / "artifacts"
        artifacts_dir.mkdir()
        
        # Run refinement cycle
        outcome = run_refinement_cycle(
            payload=payload,
            project_root=temp_project_root,
            artifacts_dir=artifacts_dir,
            llm_client=mock_llm_client,
            config=sample_config,
            apply_updates_fn=mock_apply_updates,
            re_run_pytest_fn=mock_re_run_pytest
        )
        
        # Verify success was tracked
        assert outcome.final_exit_code == 0
        mock_analyzer.mark_resolution_success.assert_called()


class TestPromptEnhancement:
    """Test enhanced prompt generation with context."""
    
    def test_prompt_loader_git_context_integration(self, temp_project_root):
        """Test PromptLoader integration with git context."""
        loader = PromptLoader()
        
        template_vars = {
            "prompt": "Fix these test failures",
            "run_id": "12345",
            "branch": "feature/fixes",
            "commit": "abc123",
            "python_version": "3.9.0",
            "platform": "Linux",
            "tests_written_count": 5,
            "last_run_command": "pytest -v",
            "failures_total": 3,
            "repo_meta": {
                "recent_changes": {
                    "time_range": "last 7 days",
                    "total_files_changed": 5,
                    "test_files_changed": 3,
                    "source_files_changed": 2,
                    "recent_commit_messages": ["Fix bug", "Add tests"]
                },
                "test_changes_detail": [
                    {"file": "test_module.py", "status": "M", "lines_added": 5, "lines_removed": 2}
                ]
            },
            "failure_analysis": {
                "pattern_frequencies": {"assertion_error": 2, "import_error": 1},
                "trending_patterns": [{"category": "assertion_error", "trend_score": 1.2}],
                "fix_suggestions": [
                    {
                        "category": "assertion_error",
                        "title": "Review assertion logic",
                        "description": "Check assertion conditions",
                        "priority": 1,
                        "automated": False,
                        "code_example": "assert x == expected"
                    }
                ]
            }
        }
        
        # Should not crash and should include context
        content = loader.get_refinement_user_content(**template_vars)
        
        assert "Fix these test failures" in content
        assert "12345" in content
        assert "feature/fixes" in content
    
    def test_prompt_context_building_methods(self, temp_project_root):
        """Test individual context building methods."""
        loader = PromptLoader()
        prompts = loader._load_yaml_file("refinement_prompts.yaml")
        
        # Test git context building
        template_vars = {
            "repo_meta": {
                "recent_changes": {
                    "time_range": "last 7 days",
                    "total_files_changed": 5,
                    "test_files_changed": 3,
                    "source_files_changed": 2,
                    "recent_commit_messages": ["Fix bug", "Add tests"]
                }
            },
            "relevant_test_changes": [
                {"file": "test_module.py", "status": "M"}
            ]
        }
        
        git_summary = loader._build_git_context_summary(template_vars, prompts)
        assert isinstance(git_summary, str)
        
        # Test pattern analysis building
        pattern_vars = {
            "failure_analysis": {
                "pattern_frequencies": {"assertion_error": 2},
                "trending_patterns": [{"category": "assertion_error", "trend_score": 1.2}],
                "fix_suggestions": [
                    {
                        "category": "assertion_error", 
                        "title": "Fix assertions",
                        "description": "Review logic",
                        "priority": 1,
                        "automated": False
                    }
                ]
            }
        }
        
        pattern_summary = loader._build_pattern_analysis_summary(pattern_vars, prompts)
        assert isinstance(pattern_summary, str)


class TestIntegrationRefinement:
    """Integration tests for the complete enhanced refinement system."""
    
    @patch('smart_test_generator.generation.refine.payload_builder.GitService')
    @patch('smart_test_generator.generation.refine.payload_builder.FailurePatternAnalyzer')
    @patch('smart_test_generator.generation.refine.refine_manager.FailurePatternAnalyzer')
    def test_complete_refinement_workflow(self, mock_refine_analyzer, mock_payload_analyzer, 
                                        mock_git_service, temp_project_root, sample_config, sample_failures):
        """Test complete workflow from payload building to refinement completion."""
        # Setup git service mock
        mock_git_service_instance = Mock()
        mock_git_service.return_value = mock_git_service_instance
        mock_git_service_instance.get_git_context.return_value = Mock(
            current_branch="main",
            current_commit="abc123",
            is_dirty=False,
            recent_changes=Mock(total_changes=0)
        )
        mock_git_service_instance.get_commit_context_for_failures.return_value = {}
        mock_git_service_instance.get_recent_test_changes.return_value = []
        
        # Setup pattern analyzer mocks
        mock_analysis = Mock()
        mock_analysis.total_failures = 4
        mock_analysis.pattern_frequencies = {FailureCategory.ASSERTION_ERROR: 2, FailureCategory.IMPORT_ERROR: 2}
        mock_analysis.trending_patterns = []
        mock_analysis.fix_suggestions = []
        mock_analysis.confidence_scores = {"test1": 0.8, "test2": 0.9}
        
        mock_payload_analyzer_instance = Mock()
        mock_payload_analyzer.return_value = mock_payload_analyzer_instance
        mock_payload_analyzer_instance.analyze_failures.return_value = mock_analysis
        mock_payload_analyzer_instance.get_success_rates.return_value = {}
        
        mock_refine_analyzer_instance = Mock()
        mock_refine_analyzer.return_value = mock_refine_analyzer_instance
        
        # Step 1: Build enhanced payload
        payload = build_payload(
            failures=sample_failures,
            project_root=temp_project_root,
            config=sample_config,
            tests_written=["test_module.py", "test_imports.py"],
            last_run_command=["python", "-m", "pytest"],
            include_git_context=True,
            include_pattern_analysis=True
        )
        
        # Verify payload has all enhancements
        assert "repo_meta" in payload
        assert "failure_analysis" in payload
        
        # Step 2: Run enhanced refinement cycle
        mock_llm_client = Mock()
        mock_llm_client.refine_tests.return_value = '{"updated_files": [{"path": "test.py", "content": "fixed"}]}'
        
        mock_apply_updates = Mock()
        mock_re_run_pytest = Mock(return_value=0)  # Success
        
        artifacts_dir = temp_project_root / "artifacts"
        artifacts_dir.mkdir()
        
        outcome = run_refinement_cycle(
            payload=payload,
            project_root=temp_project_root,
            artifacts_dir=artifacts_dir,
            llm_client=mock_llm_client,
            config=sample_config,
            apply_updates_fn=mock_apply_updates,
            re_run_pytest_fn=mock_re_run_pytest
        )
        
        # Verify complete workflow results
        assert outcome.final_exit_code == 0
        assert outcome.updated_any
        assert outcome.retry_strategy_used == "balanced"  # For mixed failure types
        assert outcome.pattern_insights is not None
        
        # Verify LLM was called with enhanced payload
        mock_llm_client.refine_tests.assert_called_once()
        call_args = mock_llm_client.refine_tests.call_args[0][0]
        assert "payload" in call_args
        assert "failure_analysis" in call_args["payload"]

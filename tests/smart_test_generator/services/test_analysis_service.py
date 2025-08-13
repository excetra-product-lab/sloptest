import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from smart_test_generator.services.analysis_service import AnalysisService
from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage, QualityAndMutationReport, TestableElement, TestQualityReport, MutationScore
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback


class TestAnalysisService:
    """Test suite for AnalysisService."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.get.return_value = []
        return config
    
    @pytest.fixture
    def mock_feedback(self):
        """Create a mock feedback object."""
        feedback = Mock(spec=UserFeedback)
        feedback.status_spinner.return_value.__enter__ = Mock()
        feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        return feedback
    
    @pytest.fixture
    def target_dir(self, tmp_path):
        """Create a temporary target directory."""
        return tmp_path / "target"
    
    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project root directory."""
        return tmp_path / "project"
    
    @pytest.fixture
    @patch('smart_test_generator.services.analysis_service.QualityAnalysisService')
    @patch('smart_test_generator.services.analysis_service.CoverageService')
    @patch('smart_test_generator.services.analysis_service.TestGenerationTracker')
    @patch('smart_test_generator.services.analysis_service.IncrementalTestGenerator')
    @patch('smart_test_generator.services.analysis_service.PythonCodebaseParser')
    def analysis_service(self, mock_parser, mock_generator, mock_tracker, mock_coverage_service, mock_quality_service, target_dir, project_root, mock_config, mock_feedback):
        """Create an AnalysisService instance with mocked dependencies."""
        return AnalysisService(target_dir, project_root, mock_config, mock_feedback)
    
    def test_init_creates_service_with_required_dependencies(self, target_dir, project_root, mock_config, mock_feedback):
        """Test that __init__ creates service with all required dependencies."""
        with patch('smart_test_generator.services.analysis_service.PythonCodebaseParser') as mock_parser, \
             patch('smart_test_generator.services.analysis_service.IncrementalTestGenerator') as mock_generator, \
             patch('smart_test_generator.services.analysis_service.TestGenerationTracker') as mock_tracker, \
             patch('smart_test_generator.services.analysis_service.CoverageService') as mock_coverage_service, \
             patch('smart_test_generator.services.analysis_service.QualityAnalysisService') as mock_quality_service:
            
            service = AnalysisService(target_dir, project_root, mock_config, mock_feedback)
            
            assert service.target_dir == target_dir
            assert service.project_root == project_root
            assert service.config == mock_config
            assert service.feedback == mock_feedback
            
            mock_parser.assert_called_once_with(str(target_dir), mock_config)
            mock_generator.assert_called_once_with(project_root, mock_config)
            mock_tracker.assert_called_once()
            mock_coverage_service.assert_called_once_with(project_root, mock_config, mock_feedback)
            mock_quality_service.assert_called_once_with(project_root, mock_config, mock_feedback)
    
    def test_init_without_feedback_creates_service_successfully(self, target_dir, project_root, mock_config):
        """Test that __init__ works without feedback parameter."""
        with patch('smart_test_generator.services.analysis_service.PythonCodebaseParser'), \
             patch('smart_test_generator.services.analysis_service.IncrementalTestGenerator'), \
             patch('smart_test_generator.services.analysis_service.TestGenerationTracker'), \
             patch('smart_test_generator.services.analysis_service.CoverageService'), \
             patch('smart_test_generator.services.analysis_service.QualityAnalysisService'):
            
            service = AnalysisService(target_dir, project_root, mock_config)
            
            assert service.target_dir == target_dir
            assert service.project_root == project_root
            assert service.config == mock_config
            assert isinstance(service.feedback, UserFeedback)
    
    def test_find_python_files_returns_parser_results(self, analysis_service):
        """Test that find_python_files returns results from parser."""
        expected_files = ['file1.py', 'file2.py', 'file3.py']
        analysis_service.parser.find_python_files.return_value = expected_files
        
        result = analysis_service.find_python_files()
        
        assert result == expected_files
        analysis_service.parser.find_python_files.assert_called_once()
    
    def test_find_python_files_returns_empty_list_when_no_files(self, analysis_service):
        """Test that find_python_files returns empty list when no files found."""
        analysis_service.parser.find_python_files.return_value = []
        
        result = analysis_service.find_python_files()
        
        assert result == []
        analysis_service.parser.find_python_files.assert_called_once()
    
    def test_analyze_files_for_generation_processes_files_needing_tests(self, analysis_service):
        """Test that analyze_files_for_generation identifies files needing test generation."""
        files = ['file1.py', 'file2.py']
        coverage_data = {
            'file1.py': TestCoverage(filepath='file1.py', line_coverage=50.0, branch_coverage=40.0, missing_lines=[], covered_functions=set(), uncovered_functions=set()),
            'file2.py': TestCoverage(filepath='file2.py', line_coverage=90.0, branch_coverage=85.0, missing_lines=[], covered_functions=set(), uncovered_functions=set())
        }
        
        analysis_service.coverage_service.analyze_coverage.return_value = coverage_data
        analysis_service.tracker.should_generate_tests.side_effect = [
            (True, 'Low coverage'),
            (False, 'Adequate coverage')
        ]
        
        files_to_process, reasons, returned_coverage = analysis_service.analyze_files_for_generation(files)
        
        assert files_to_process == ['file1.py']
        assert reasons == {'file1.py': 'Low coverage (Note: Found 0 existing test file(s): [])'}
        assert returned_coverage == coverage_data
        analysis_service.coverage_service.analyze_coverage.assert_called_once_with(files)
    
    def test_analyze_files_for_generation_with_force_flag(self, analysis_service):
        """Test that analyze_files_for_generation respects force flag."""
        files = ['file1.py']
        coverage_data = {'file1.py': TestCoverage(filepath='file1.py', line_coverage=95.0, branch_coverage=90.0, missing_lines=[], covered_functions=set(), uncovered_functions=set())}
        
        analysis_service.coverage_service.analyze_coverage.return_value = coverage_data
        analysis_service.tracker.should_generate_tests.return_value = (True, 'Forced generation')
        
        files_to_process, reasons, returned_coverage = analysis_service.analyze_files_for_generation(files, force=True)
        
        assert files_to_process == ['file1.py']
        assert reasons == {'file1.py': 'Forced generation (Note: Found 0 existing test file(s): [])'}
        analysis_service.tracker.should_generate_tests.assert_called_once_with(
            'file1.py', coverage_data['file1.py'], analysis_service.config, True
        )
    
    def test_analyze_files_for_generation_returns_empty_when_no_files_need_tests(self, analysis_service):
        """Test that analyze_files_for_generation returns empty results when no files need tests."""
        files = ['file1.py', 'file2.py']
        coverage_data = {
            'file1.py': TestCoverage(filepath='file1.py', line_coverage=95.0, branch_coverage=90.0, missing_lines=[], covered_functions=set(), uncovered_functions=set()),
            'file2.py': TestCoverage(filepath='file2.py', line_coverage=90.0, branch_coverage=85.0, missing_lines=[], covered_functions=set(), uncovered_functions=set())
        }
        
        analysis_service.coverage_service.analyze_coverage.return_value = coverage_data
        analysis_service.tracker.should_generate_tests.return_value = (False, 'Adequate coverage')
        
        files_to_process, reasons, returned_coverage = analysis_service.analyze_files_for_generation(files)
        
        assert files_to_process == []
        assert reasons == {}
        assert returned_coverage == coverage_data
    
    def test_create_test_plans_generates_plans_for_files(self, analysis_service):
        """Test that create_test_plans generates test plans for specified files."""
        files = ['file1.py', 'file2.py']
        coverage_data = {
            'file1.py': TestCoverage(filepath='file1.py', line_coverage=50.0, branch_coverage=40.0, missing_lines=[], covered_functions=set(), uncovered_functions=set()),
            'file2.py': TestCoverage(filepath='file2.py', line_coverage=70.0, branch_coverage=65.0, missing_lines=[], covered_functions=set(), uncovered_functions=set())
        }
        
        plan1 = TestGenerationPlan(
            source_file='file1.py', 
            existing_test_files=['test_file1.py'], 
            elements_to_test=[
                TestableElement(name='func1', type='function', filepath='file1.py', line_number=1, signature='def func1():'),
                TestableElement(name='func2', type='function', filepath='file1.py', line_number=2, signature='def func2():')
            ],
            coverage_before=None,
            estimated_coverage_after=80.0
        )
        plan2 = TestGenerationPlan(
            source_file='file2.py', 
            existing_test_files=['test_file2.py'], 
            elements_to_test=[
                TestableElement(name='func3', type='function', filepath='file2.py', line_number=1, signature='def func3():')
            ],
            coverage_before=None,
            estimated_coverage_after=75.0
        )
        
        analysis_service.test_generator.generate_test_plan.side_effect = [plan1, plan2]
        analysis_service.quality_service.update_test_plans_with_quality_insights.return_value = [plan1, plan2]
        
        result = analysis_service.create_test_plans(files, coverage_data)
        
        assert result == [plan1, plan2]
        assert analysis_service.test_generator.generate_test_plan.call_count == 2
        analysis_service.quality_service.update_test_plans_with_quality_insights.assert_called_once_with([plan1, plan2])
    
    def test_create_test_plans_filters_empty_plans(self, analysis_service):
        """Test that create_test_plans filters out plans with no elements to test."""
        files = ['file1.py', 'file2.py']
        coverage_data = {
            'file1.py': TestCoverage(filepath='file1.py', line_coverage=50.0, branch_coverage=40.0, missing_lines=[], covered_functions=set(), uncovered_functions=set()),
            'file2.py': TestCoverage(filepath='file2.py', line_coverage=70.0, branch_coverage=65.0, missing_lines=[], covered_functions=set(), uncovered_functions=set())
        }
        
        plan1 = TestGenerationPlan(
            source_file='file1.py', 
            existing_test_files=['test_file1.py'], 
            elements_to_test=[
                TestableElement(name='func1', type='function', filepath='file1.py', line_number=1, signature='def func1():')
            ],
            coverage_before=None,
            estimated_coverage_after=75.0
        )
        plan2 = TestGenerationPlan(
            source_file='file2.py', 
            existing_test_files=['test_file2.py'], 
            elements_to_test=[],
            coverage_before=None,
            estimated_coverage_after=0.0
        )
        
        analysis_service.test_generator.generate_test_plan.side_effect = [plan1, plan2]
        analysis_service.quality_service.update_test_plans_with_quality_insights.return_value = [plan1]
        
        result = analysis_service.create_test_plans(files, coverage_data)
        
        assert result == [plan1]
        analysis_service.quality_service.update_test_plans_with_quality_insights.assert_called_once_with([plan1])
    
    def test_create_test_plans_skips_quality_enhancement_when_no_plans(self, analysis_service):
        """Test that create_test_plans skips quality enhancement when no valid plans exist."""
        files = ['file1.py']
        coverage_data = {'file1.py': TestCoverage(filepath='file1.py', line_coverage=50.0, branch_coverage=40.0, missing_lines=[], covered_functions=set(), uncovered_functions=set())}
        
        empty_plan = TestGenerationPlan(
            source_file='file1.py', 
            existing_test_files=['test_file1.py'], 
            elements_to_test=[],
            coverage_before=None,
            estimated_coverage_after=0.0
        )
        analysis_service.test_generator.generate_test_plan.return_value = empty_plan
        
        result = analysis_service.create_test_plans(files, coverage_data)
        
        assert result == []
        analysis_service.quality_service.update_test_plans_with_quality_insights.assert_not_called()
    
    def test_analyze_test_quality_returns_quality_reports(self, analysis_service):
        """Test that analyze_test_quality returns quality reports for test plans."""
        test_plans = [
            TestGenerationPlan(
                source_file='file1.py', 
                existing_test_files=['test_file1.py'], 
                elements_to_test=[
                    TestableElement(name='func1', type='function', filepath='file1.py', line_number=1, signature='def func1():')
                ],
                coverage_before=None,
                estimated_coverage_after=75.0
            ),
            TestGenerationPlan(
                source_file='file2.py', 
                existing_test_files=['test_file2.py'], 
                elements_to_test=[
                    TestableElement(name='func2', type='function', filepath='file2.py', line_number=1, signature='def func2():')
                ],
                coverage_before=None,
                estimated_coverage_after=75.0
            )
        ]
        
        # Create mock reports for testing
        from smart_test_generator.models.data_models import TestQualityReport, MutationScore
        
        mock_quality_report1 = TestQualityReport(test_file='test_file1.py', overall_score=75.0)
        mock_mutation_score1 = MutationScore(
            source_file='file1.py', test_files=['test_file1.py'], 
            total_mutants=10, killed_mutants=8, surviving_mutants=2, 
            timeout_mutants=0, error_mutants=0, mutation_score=80.0, mutation_score_effective=80.0
        )
        
        mock_quality_report2 = TestQualityReport(test_file='test_file2.py', overall_score=70.0)
        mock_mutation_score2 = MutationScore(
            source_file='file2.py', test_files=['test_file2.py'], 
            total_mutants=10, killed_mutants=7, surviving_mutants=3, 
            timeout_mutants=0, error_mutants=0, mutation_score=70.0, mutation_score_effective=70.0
        )
        
        expected_reports = {
            'file1.py': QualityAndMutationReport(
                source_file='file1.py', 
                test_files=['test_file1.py'],
                quality_report=mock_quality_report1,
                mutation_score=mock_mutation_score1,
                overall_rating='good',
                confidence_level=0.8
            ),
            'file2.py': QualityAndMutationReport(
                source_file='file2.py', 
                test_files=['test_file2.py'],
                quality_report=mock_quality_report2,
                mutation_score=mock_mutation_score2,
                overall_rating='fair',
                confidence_level=0.7
            )
        }
        
        analysis_service.quality_service.analyze_test_plans_quality.return_value = expected_reports
        
        result = analysis_service.analyze_test_quality(test_plans)
        
        assert result == expected_reports
        analysis_service.quality_service.analyze_test_plans_quality.assert_called_once_with(test_plans)
    
    def test_analyze_test_quality_handles_empty_test_plans(self, analysis_service):
        """Test that analyze_test_quality handles empty test plans list."""
        test_plans = []
        expected_reports = {}
        
        result = analysis_service.analyze_test_quality(test_plans)
        
        assert result == expected_reports
        # Method should not be called when test_plans is empty
        analysis_service.quality_service.analyze_test_plans_quality.assert_not_called()
    
    def test_generate_quality_gaps_report_creates_formatted_report(self, analysis_service):
        """Test that generate_quality_gaps_report creates a formatted report with gaps."""
        quality_reports = {
            'file1.py': QualityAndMutationReport(
                source_file='file1.py',
                test_files=['test_file1.py'],
                quality_report=TestQualityReport(test_file='test_file1.py', overall_score=80.0),
                mutation_score=MutationScore(
                    source_file='file1.py', test_files=['test_file1.py'], total_mutants=10, killed_mutants=8,
                    surviving_mutants=2, timeout_mutants=0, error_mutants=0, mutation_score=80.0,
                    mutation_score_effective=80.0, mutant_results=[], weak_spots=[], mutation_distribution={},
                    total_execution_time=0.0, average_test_time=0.0
                ),
                overall_rating='good',
                confidence_level=0.8
            ),
            'file2.py': QualityAndMutationReport(
                source_file='file2.py',
                test_files=['test_file2.py'],
                quality_report=TestQualityReport(test_file='test_file2.py', overall_score=75.0),
                mutation_score=MutationScore(
                    source_file='file2.py', test_files=['test_file2.py'], total_mutants=8, killed_mutants=6,
                    surviving_mutants=2, timeout_mutants=0, error_mutants=0, mutation_score=75.0,
                    mutation_score_effective=75.0, mutant_results=[], weak_spots=[], mutation_distribution={},
                    total_execution_time=0.0, average_test_time=0.0
                ),
                overall_rating='fair',
                confidence_level=0.7
            )
        }
        
        gaps = {
            'file1.py': ['Missing edge case tests', 'Low assertion coverage'],
            'file2.py': ['No error condition tests']
        }
        
        analysis_service.quality_service.identify_quality_gaps.return_value = gaps
        
        result = analysis_service.generate_quality_gaps_report(quality_reports)
        
        assert 'Test Quality Analysis' in result
        assert '📁 file1.py' in result
        assert '⚠️  Missing edge case tests' in result
        assert '⚠️  Low assertion coverage' in result
        assert '📁 file2.py' in result
        assert '⚠️  No error condition tests' in result
        analysis_service.quality_service.identify_quality_gaps.assert_called_once_with(quality_reports)
    
    def test_generate_quality_gaps_report_returns_success_message_when_no_gaps(self, analysis_service):
        """Test that generate_quality_gaps_report returns success message when no gaps found."""
        quality_reports = {
            'file1.py': QualityAndMutationReport(
                source_file='file1.py',
                test_files=['test_file1.py'],
                quality_report=TestQualityReport(test_file='test_file1.py', overall_score=95.0),
                mutation_score=MutationScore(
                    source_file='file1.py', test_files=['test_file1.py'], total_mutants=10, killed_mutants=10,
                    surviving_mutants=0, timeout_mutants=0, error_mutants=0, mutation_score=100.0,
                    mutation_score_effective=100.0, mutant_results=[], weak_spots=[], mutation_distribution={},
                    total_execution_time=0.0, average_test_time=0.0
                ),
                overall_rating='excellent',
                confidence_level=0.95
            )
        }
        
        analysis_service.quality_service.identify_quality_gaps.return_value = {}
        
        result = analysis_service.generate_quality_gaps_report(quality_reports)
        
        assert result == '✅ No significant quality gaps found in existing tests!'
        analysis_service.quality_service.identify_quality_gaps.assert_called_once_with(quality_reports)
    
    def test_generate_analysis_report_creates_comprehensive_report(self, analysis_service, project_root):
        """Test that generate_analysis_report creates a comprehensive analysis report."""
        all_files = ['file1.py', 'file2.py', 'file3.py']
        files_to_process = ['file1.py', 'file2.py']
        test_plans = [
            TestGenerationPlan(
                source_file='file1.py', 
                existing_test_files=['test_file1.py'], 
                elements_to_test=[
                    TestableElement(name='func1', type='function', filepath='file1.py', line_number=1, signature='def func1():'),
                    TestableElement(name='func2', type='function', filepath='file1.py', line_number=2, signature='def func2():')
                ],
                coverage_before=None,
                estimated_coverage_after=80.0
            ),
            TestGenerationPlan(
                source_file='file2.py', 
                existing_test_files=['test_file2.py'], 
                elements_to_test=[
                    TestableElement(name='func3', type='function', filepath='file2.py', line_number=1, signature='def func3():')
                ],
                coverage_before=None,
                estimated_coverage_after=75.0
            )
        ]
        reasons = {
            'file1.py': 'Low coverage',
            'file2.py': 'Missing tests'
        }
        
        analysis_service.project_root = project_root
        
        result = analysis_service.generate_analysis_report(all_files, files_to_process, test_plans, reasons)
        
        assert 'Analysis Report' in result
        assert 'Total Python files: 3' in result
        assert 'Files needing test generation: 2' in result
        assert 'Files to process:' in result
        assert 'Low coverage' in result
        assert 'Missing tests' in result
        assert 'Total elements needing tests: 3' in result
    
    def test_generate_analysis_report_handles_no_files_to_process(self, analysis_service):
        """Test that generate_analysis_report handles case with no files to process."""
        all_files = ['file1.py', 'file2.py']
        files_to_process = []
        test_plans = []
        reasons = {}
        
        result = analysis_service.generate_analysis_report(all_files, files_to_process, test_plans, reasons)
        
        assert 'Analysis Report' in result
        assert 'Total Python files: 2' in result
        assert 'Files needing test generation: 0' in result
        assert 'All files have adequate test coverage!' in result
    
    def test_get_generation_status_returns_formatted_status(self, analysis_service):
        """Test that get_generation_status returns formatted status with generation history."""
        generation_log = [
            {
                'timestamp': '2023-01-01 10:00:00',
                'filepath': 'file1.py',
                'reason': 'Low coverage',
                'elements_generated': 5,
                'coverage_before': 45.0,
                'coverage_after': 78.0
            },
            {
                'timestamp': '2023-01-01 11:00:00',
                'filepath': 'file2.py',
                'reason': 'Missing tests',
                'elements_generated': 3,
                'coverage_before': 20.0,
                'coverage_after': 65.0
            }
        ]
        
        analysis_service.tracker.state.generation_log = generation_log
        
        result = analysis_service.get_generation_status()
        
        assert 'Test Generation Status' in result
        assert '2023-01-01 10:00:00:' in result
        assert 'File: file1.py' in result
        assert 'Reason: Low coverage' in result
        assert 'Elements generated: 5' in result
        assert 'Coverage: 45.0% → 78.0%' in result
        assert '2023-01-01 11:00:00:' in result
        assert 'File: file2.py' in result
    
    def test_get_generation_status_returns_no_history_message_when_empty(self, analysis_service):
        """Test that get_generation_status returns no history message when log is empty."""
        analysis_service.tracker.state.generation_log = []
        
        result = analysis_service.get_generation_status()
        
        assert result == 'No test generation history found.'
    
    def test_get_generation_status_limits_to_last_ten_entries(self, analysis_service):
        """Test that get_generation_status limits output to last 10 entries."""
        # Create 15 log entries
        generation_log = []
        for i in range(15):
            generation_log.append({
                'timestamp': f'2023-01-01 {i:02d}:00:00',
                'filepath': f'file{i}.py',
                'reason': 'Test reason',
                'elements_generated': 1,
                'coverage_before': 50.0,
                'coverage_after': 75.0
            })
        
        analysis_service.tracker.state.generation_log = generation_log
        
        result = analysis_service.get_generation_status()
        
        # Should only contain the last 10 entries (5-14)
        assert 'file5.py' in result
        assert 'file14.py' in result
        assert 'file4.py' not in result
        assert 'file0.py' not in result
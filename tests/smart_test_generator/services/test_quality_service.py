import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict

from smart_test_generator.services.quality_service import QualityAnalysisService
from smart_test_generator.models.data_models import (
    TestQualityReport, MutationScore, QualityAndMutationReport,
    TestGenerationPlan, WeakSpot
)
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback


class TestQualityAnalysisService:
    """Test suite for QualityAnalysisService."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.get.return_value = None
        return config
    
    @pytest.fixture
    def mock_feedback(self):
        """Create a mock feedback object."""
        return Mock(spec=UserFeedback)
    
    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project root."""
        return tmp_path
    
    @pytest.fixture
    def quality_service(self, project_root, mock_config, mock_feedback):
        """Create a QualityAnalysisService instance."""
        with patch('smart_test_generator.services.quality_service.TestQualityEngine'), \
             patch('smart_test_generator.services.quality_service.MutationTestingEngine'), \
             patch('smart_test_generator.services.quality_service.get_python_quality_analyzers'), \
             patch('smart_test_generator.services.quality_service.get_python_mutation_operators'):
            return QualityAnalysisService(project_root, mock_config, mock_feedback)
    
    def test_init_with_default_config_values(self, project_root, mock_config, mock_feedback):
        """Test initialization with default configuration values."""
        mock_config.get.side_effect = lambda key, default: {
            'quality.mutation_timeout': 30,
            'quality.max_mutants_per_file': 50,
            'quality.minimum_quality_score': 75.0,
            'quality.minimum_mutation_score': 80.0,
            'quality.enable_mutation_testing': True
        }.get(key, default)
        
        with patch('smart_test_generator.services.quality_service.TestQualityEngine') as mock_quality_engine, \
             patch('smart_test_generator.services.quality_service.MutationTestingEngine') as mock_mutation_engine, \
             patch('smart_test_generator.services.quality_service.get_python_quality_analyzers') as mock_analyzers, \
             patch('smart_test_generator.services.quality_service.get_python_mutation_operators') as mock_operators:
            
            mock_analyzers.return_value = []
            mock_operators.return_value = []
            
            service = QualityAnalysisService(project_root, mock_config, mock_feedback)
            
            assert service.max_mutants_per_file == 50
            assert service.quality_threshold == 75.0
            assert service.mutation_threshold == 80.0
            assert service.enable_mutation_testing is True
            mock_quality_engine.assert_called_once_with(custom_analyzers=[])
            mock_mutation_engine.assert_called_once_with(operators=[], timeout=30)
    
    def test_init_with_custom_config_values(self, project_root, mock_feedback):
        """Test initialization with custom configuration values."""
        mock_config = Mock(spec=Config)
        mock_config.get.side_effect = lambda key, default: {
            'quality.mutation_timeout': 60,
            'quality.max_mutants_per_file': 100,
            'quality.minimum_quality_score': 85.0,
            'quality.minimum_mutation_score': 90.0,
            'quality.enable_mutation_testing': False
        }.get(key, default)
        
        with patch('smart_test_generator.services.quality_service.TestQualityEngine'), \
             patch('smart_test_generator.services.quality_service.MutationTestingEngine') as mock_mutation_engine, \
             patch('smart_test_generator.services.quality_service.get_python_quality_analyzers'), \
             patch('smart_test_generator.services.quality_service.get_python_mutation_operators') as mock_operators:
            
            mock_operators.return_value = []
            
            service = QualityAnalysisService(project_root, mock_config, mock_feedback)
            
            assert service.max_mutants_per_file == 100
            assert service.quality_threshold == 85.0
            assert service.mutation_threshold == 90.0
            assert service.enable_mutation_testing is False
            mock_mutation_engine.assert_called_once_with(operators=[], timeout=60)
    
    def test_init_without_feedback(self, project_root, mock_config):
        """Test initialization without feedback object."""
        with patch('smart_test_generator.services.quality_service.TestQualityEngine'), \
             patch('smart_test_generator.services.quality_service.MutationTestingEngine'), \
             patch('smart_test_generator.services.quality_service.get_python_quality_analyzers'), \
             patch('smart_test_generator.services.quality_service.get_python_mutation_operators'):
            
            service = QualityAnalysisService(project_root, mock_config)
            assert service.feedback is None
    
    def test_analyze_test_quality_success(self, quality_service):
        """Test successful test quality analysis."""
        test_file = "test_example.py"
        source_file = "example.py"
        expected_report = TestQualityReport(
            test_file=test_file,
            overall_score=85.0,
            improvement_suggestions=["Add more edge cases"],
            priority_fixes=["Fix weak assertions"]
        )
        
        quality_service.quality_engine.analyze_test_quality.return_value = expected_report
        
        result = quality_service.analyze_test_quality(test_file, source_file)
        
        assert result == expected_report
        quality_service.quality_engine.analyze_test_quality.assert_called_once_with(test_file, source_file)
    
    def test_analyze_test_quality_with_exception(self, quality_service):
        """Test test quality analysis when exception occurs."""
        test_file = "test_example.py"
        source_file = "example.py"
        
        quality_service.quality_engine.analyze_test_quality.side_effect = Exception("Analysis failed")
        
        result = quality_service.analyze_test_quality(test_file, source_file)
        
        assert result.test_file == test_file
        assert result.overall_score == 0.0
        assert "Unable to analyze test quality" in result.improvement_suggestions
        assert "Check file exists and is readable" in result.priority_fixes
    
    def test_analyze_test_quality_without_source_file(self, quality_service):
        """Test test quality analysis without source file."""
        test_file = "test_example.py"
        expected_report = TestQualityReport(
            test_file=test_file,
            overall_score=75.0,
            improvement_suggestions=[],
            priority_fixes=[]
        )
        
        quality_service.quality_engine.analyze_test_quality.return_value = expected_report
        
        result = quality_service.analyze_test_quality(test_file)
        
        assert result == expected_report
        quality_service.quality_engine.analyze_test_quality.assert_called_once_with(test_file, "")
    
    def test_run_mutation_testing_success(self, quality_service):
        """Test successful mutation testing."""
        source_file = "example.py"
        test_files = ["test_example.py"]
        expected_score = MutationScore(
            source_file=source_file,
            test_files=test_files,
            total_mutants=10,
            killed_mutants=8,
            surviving_mutants=2,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=80.0,
            mutation_score_effective=80.0
        )
        
        quality_service.mutation_engine.run_mutation_testing.return_value = expected_score
        
        result = quality_service.run_mutation_testing(source_file, test_files)
        
        assert result == expected_score
        quality_service.mutation_engine.run_mutation_testing.assert_called_once_with(
            source_file, test_files, max_mutants=quality_service.max_mutants_per_file
        )
    
    def test_run_mutation_testing_disabled(self, quality_service):
        """Test mutation testing when disabled in configuration."""
        quality_service.enable_mutation_testing = False
        source_file = "example.py"
        test_files = ["test_example.py"]
        
        result = quality_service.run_mutation_testing(source_file, test_files)
        
        assert result is None
        quality_service.mutation_engine.run_mutation_testing.assert_not_called()
    
    def test_run_mutation_testing_with_exception(self, quality_service):
        """Test mutation testing when exception occurs."""
        source_file = "example.py"
        test_files = ["test_example.py"]
        
        quality_service.mutation_engine.run_mutation_testing.side_effect = Exception("Mutation failed")
        
        result = quality_service.run_mutation_testing(source_file, test_files)
        
        assert result is None
    
    def test_comprehensive_quality_analysis_with_mutation_testing(self, quality_service):
        """Test comprehensive quality analysis with mutation testing enabled."""
        source_file = "example.py"
        test_files = ["test_example.py"]
        
        quality_report = TestQualityReport(
            test_file=test_files[0],
            overall_score=80.0,
            improvement_suggestions=["Add edge cases"],
            priority_fixes=["Fix assertions"]
        )
        
        mutation_score = MutationScore(
            source_file=source_file,
            test_files=test_files,
            total_mutants=10,
            killed_mutants=8,
            surviving_mutants=2,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=80.0,
            mutation_score_effective=80.0
        )
        
        with patch.object(quality_service, '_analyze_combined_test_quality', return_value=quality_report), \
             patch.object(quality_service, 'run_mutation_testing', return_value=mutation_score), \
             patch.object(quality_service, '_create_combined_report') as mock_create_report:
            
            expected_report = QualityAndMutationReport(
                source_file=source_file,
                test_files=test_files,
                quality_report=quality_report,
                mutation_score=mutation_score,
                overall_rating="good",
                confidence_level=0.8
            )
            mock_create_report.return_value = expected_report
            
            result = quality_service.comprehensive_quality_analysis(source_file, test_files)
            
            assert result == expected_report
            mock_create_report.assert_called_once_with(source_file, test_files, quality_report, mutation_score)
    
    def test_comprehensive_quality_analysis_without_mutation_testing(self, quality_service):
        """Test comprehensive quality analysis with mutation testing disabled."""
        quality_service.enable_mutation_testing = False
        source_file = "example.py"
        test_files = ["test_example.py"]
        
        quality_report = TestQualityReport(
            test_file=test_files[0],
            overall_score=80.0,
            improvement_suggestions=[],
            priority_fixes=[]
        )
        
        with patch.object(quality_service, '_analyze_combined_test_quality', return_value=quality_report), \
             patch.object(quality_service, '_create_empty_mutation_score') as mock_empty_score, \
             patch.object(quality_service, '_create_combined_report') as mock_create_report:
            
            empty_mutation_score = MutationScore(
                source_file=source_file,
                test_files=test_files,
                total_mutants=0,
                killed_mutants=0,
                surviving_mutants=0,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=0.0,
                mutation_score_effective=0.0
            )
            mock_empty_score.return_value = empty_mutation_score
            
            expected_report = QualityAndMutationReport(
                source_file=source_file,
                test_files=test_files,
                quality_report=quality_report,
                mutation_score=empty_mutation_score,
                overall_rating="fair",
                confidence_level=0.6
            )
            mock_create_report.return_value = expected_report
            
            result = quality_service.comprehensive_quality_analysis(source_file, test_files)
            
            assert result == expected_report
            mock_empty_score.assert_called_once_with(source_file, test_files)
    
    def test_comprehensive_quality_analysis_with_empty_test_files(self, quality_service):
        """Test comprehensive quality analysis with empty test files list."""
        source_file = "example.py"
        test_files = []
        
        quality_report = TestQualityReport(
            test_file="no_test_files",
            overall_score=0.0,
            improvement_suggestions=["Unable to analyze test quality"],
            priority_fixes=["Check file exists and is readable"]
        )
        
        with patch.object(quality_service, '_analyze_combined_test_quality', return_value=quality_report), \
             patch.object(quality_service, '_create_empty_mutation_score') as mock_empty_score, \
             patch.object(quality_service, '_create_combined_report') as mock_create_report:
            
            empty_mutation_score = MutationScore(
                source_file=source_file,
                test_files=test_files,
                total_mutants=0,
                killed_mutants=0,
                surviving_mutants=0,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=0.0,
                mutation_score_effective=0.0
            )
            mock_empty_score.return_value = empty_mutation_score
            
            expected_report = QualityAndMutationReport(
                source_file=source_file,
                test_files=test_files,
                quality_report=quality_report,
                mutation_score=empty_mutation_score,
                overall_rating="poor",
                confidence_level=0.0
            )
            mock_create_report.return_value = expected_report
            
            result = quality_service.comprehensive_quality_analysis(source_file, test_files)
            
            assert result == expected_report
    
    def test_analyze_test_plans_quality_with_existing_tests(self, quality_service):
        """Test analyzing quality for test plans with existing test files."""
        plan1 = TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=["test_file1.py"],
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        plan2 = TestGenerationPlan(
            source_file="file2.py",
            existing_test_files=["test_file2.py"],
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        plan3 = TestGenerationPlan(
            source_file="file3.py",
            existing_test_files=[],  # No existing tests
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        test_plans = [plan1, plan2, plan3]
        
        report1 = QualityAndMutationReport(
            source_file="file1.py",
            test_files=["test_file1.py"],
            quality_report=TestQualityReport(
                test_file="test_file1.py",
                overall_score=85.0,
                improvement_suggestions=[],
                priority_fixes=[]
            ),
            mutation_score=MutationScore(
                source_file="file1.py",
                test_files=["test_file1.py"],
                total_mutants=10,
                killed_mutants=8,
                surviving_mutants=2,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=80.0,
                mutation_score_effective=80.0
            ),
            overall_rating="good",
            confidence_level=0.8
        )
        
        report2 = QualityAndMutationReport(
            source_file="file2.py",
            test_files=["test_file2.py"],
            quality_report=TestQualityReport(
                test_file="test_file2.py",
                overall_score=75.0,
                improvement_suggestions=[],
                priority_fixes=[]
            ),
            mutation_score=MutationScore(
                source_file="file2.py",
                test_files=["test_file2.py"],
                total_mutants=10,
                killed_mutants=7,
                surviving_mutants=3,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=70.0,
                mutation_score_effective=70.0
            ),
            overall_rating="fair",
            confidence_level=0.7
        )
        
        with patch.object(quality_service, 'comprehensive_quality_analysis', side_effect=[report1, report2]):
            result = quality_service.analyze_test_plans_quality(test_plans)
            
            assert len(result) == 2
            assert result["file1.py"] == report1
            assert result["file2.py"] == report2
            assert "file3.py" not in result
    
    def test_analyze_test_plans_quality_with_no_existing_tests(self, quality_service):
        """Test analyzing quality for test plans with no existing test files."""
        test_plans = [
            TestGenerationPlan(
                source_file="file1.py",
                existing_test_files=[],
                missing_test_elements=[],
                quality_score_target=80.0,
                mutation_score_target=85.0
            )
        ]
        
        result = quality_service.analyze_test_plans_quality(test_plans)
        
        assert result == {}
    
    def test_identify_quality_gaps_with_low_scores(self, quality_service):
        """Test identifying quality gaps with low quality and mutation scores."""
        quality_service.quality_threshold = 75.0
        quality_service.mutation_threshold = 80.0
        
        weak_spot = WeakSpot(
            line_number=10,
            description="Uncovered branch condition",
            severity="critical",
            suggested_tests=["Test edge case for condition"]
        )
        
        quality_reports = {
            "file1.py": QualityAndMutationReport(
                source_file="file1.py",
                test_files=["test_file1.py"],
                quality_report=TestQualityReport(
                    test_file="test_file1.py",
                    overall_score=60.0,  # Below threshold
                    improvement_suggestions=[],
                    priority_fixes=["Add more assertions", "Improve test structure"]
                ),
                mutation_score=MutationScore(
                    source_file="file1.py",
                    test_files=["test_file1.py"],
                    total_mutants=10,
                    killed_mutants=6,
                    surviving_mutants=4,
                    timeout_mutants=0,
                    error_mutants=0,
                    mutation_score=60.0,  # Below threshold
                    mutation_score_effective=60.0,
                    weak_spots=[weak_spot]
                ),
                overall_rating="poor",
                confidence_level=0.6
            )
        }
        
        with patch.object(quality_reports["file1.py"].mutation_score, 'get_weak_spots_by_severity', return_value=[weak_spot]):
            result = quality_service.identify_quality_gaps(quality_reports)
            
            assert "file1.py" in result
            gaps = result["file1.py"]
            assert any("Quality score (60.0) below threshold (75.0)" in gap for gap in gaps)
            assert any("Mutation score (60.0%) below threshold (80.0%)" in gap for gap in gaps)
            assert any("Add more assertions" in gap for gap in gaps)
            assert any("Improve test structure" in gap for gap in gaps)
            assert any("Critical weak spot: Uncovered branch condition" in gap for gap in gaps)
    
    def test_identify_quality_gaps_with_high_scores(self, quality_service):
        """Test identifying quality gaps with high quality and mutation scores."""
        quality_service.quality_threshold = 75.0
        quality_service.mutation_threshold = 80.0
        
        quality_reports = {
            "file1.py": QualityAndMutationReport(
                source_file="file1.py",
                test_files=["test_file1.py"],
                quality_report=TestQualityReport(
                    test_file="test_file1.py",
                    overall_score=85.0,  # Above threshold
                    improvement_suggestions=[],
                    priority_fixes=[]  # No priority fixes
                ),
                mutation_score=MutationScore(
                    source_file="file1.py",
                    test_files=["test_file1.py"],
                    total_mutants=10,
                    killed_mutants=9,
                    surviving_mutants=1,
                    timeout_mutants=0,
                    error_mutants=0,
                    mutation_score=90.0,  # Above threshold
                    mutation_score_effective=90.0,
                    weak_spots=[]
                ),
                overall_rating="excellent",
                confidence_level=0.9
            )
        }
        
        with patch.object(quality_reports["file1.py"].mutation_score, 'get_weak_spots_by_severity', return_value=[]):
            result = quality_service.identify_quality_gaps(quality_reports)
            
            assert result == {}
    
    def test_identify_quality_gaps_with_empty_reports(self, quality_service):
        """Test identifying quality gaps with empty quality reports."""
        result = quality_service.identify_quality_gaps({})
        assert result == {}
    
    def test_generate_quality_improvement_suggestions_with_low_scores(self, quality_service):
        """Test generating quality improvement suggestions for low scores."""
        critical_weak_spot = WeakSpot(
            line_number=10,
            description="Critical uncovered branch",
            severity="critical",
            suggested_tests=["Test critical edge case"]
        )
        
        high_weak_spot = WeakSpot(
            line_number=20,
            description="High priority uncovered condition",
            severity="high",
            suggested_tests=["Test high priority case", "Test another high priority case"]
        )
        
        report = QualityAndMutationReport(
            source_file="file1.py",
            test_files=["test_file1.py"],
            quality_report=TestQualityReport(
                test_file="test_file1.py",
                overall_score=45.0,  # Very low score
                improvement_suggestions=[],
                priority_fixes=["Fix weak assertions", "Add edge cases", "Improve structure"]
            ),
            mutation_score=MutationScore(
                source_file="file1.py",
                test_files=["test_file1.py"],
                total_mutants=10,
                killed_mutants=3,
                surviving_mutants=7,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=30.0,  # Very low score
                mutation_score_effective=30.0,
                weak_spots=[critical_weak_spot, high_weak_spot]
            ),
            overall_rating="poor",
            confidence_level=0.3
        )
        
        with patch.object(report.mutation_score, 'get_weak_spots_by_severity') as mock_get_weak_spots:
            mock_get_weak_spots.side_effect = lambda severity: {
                "critical": [critical_weak_spot],
                "high": [high_weak_spot]
            }.get(severity, [])
            
            result = quality_service.generate_quality_improvement_suggestions(report)
            
            # Should include priority fixes from quality analysis
            assert "Fix weak assertions" in result
            assert "Add edge cases" in result
            assert "Improve structure" in result
            
            # Should include suggestions from weak spots
            assert "Test critical edge case" in result
            assert "Test high priority case" in result
            
            # Should include additional suggestions for very low scores
            assert "Focus on basic test structure and assertions before advanced techniques" in result
            assert "Add more comprehensive test scenarios to catch basic bugs" in result
    
    def test_generate_quality_improvement_suggestions_with_high_scores(self, quality_service):
        """Test generating quality improvement suggestions for high scores."""
        report = QualityAndMutationReport(
            source_file="file1.py",
            test_files=["test_file1.py"],
            quality_report=TestQualityReport(
                test_file="test_file1.py",
                overall_score=85.0,  # High score
                improvement_suggestions=[],
                priority_fixes=["Minor improvement"]
            ),
            mutation_score=MutationScore(
                source_file="file1.py",
                test_files=["test_file1.py"],
                total_mutants=10,
                killed_mutants=9,
                surviving_mutants=1,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=90.0,  # High score
                mutation_score_effective=90.0,
                weak_spots=[]
            ),
            overall_rating="excellent",
            confidence_level=0.9
        )
        
        with patch.object(report.mutation_score, 'get_weak_spots_by_severity', return_value=[]):
            result = quality_service.generate_quality_improvement_suggestions(report)
            
            # Should only include the minor improvement
            assert "Minor improvement" in result
            # Should not include additional suggestions for high scores
            assert "Focus on basic test structure and assertions before advanced techniques" not in result
            assert "Add more comprehensive test scenarios to catch basic bugs" not in result
    
    def test_update_test_plans_with_quality_insights_with_existing_tests(self, quality_service):
        """Test updating test plans with quality insights for plans with existing tests."""
        original_plan = TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=["test_file1.py"],
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        weak_spot = WeakSpot(
            line_number=10,
            description="Uncovered condition",
            severity="high",
            suggested_tests=["Test edge case"]
        )
        
        quality_report = QualityAndMutationReport(
            source_file="file1.py",
            test_files=["test_file1.py"],
            quality_report=TestQualityReport(
                test_file="test_file1.py",
                overall_score=65.0,  # Moderate score
                improvement_suggestions=[],
                priority_fixes=[]
            ),
            mutation_score=MutationScore(
                source_file="file1.py",
                test_files=["test_file1.py"],
                total_mutants=10,
                killed_mutants=7,
                surviving_mutants=3,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=70.0,  # Moderate score
                mutation_score_effective=70.0,
                weak_spots=[weak_spot]
            ),
            overall_rating="fair",
            confidence_level=0.7
        )
        
        with patch.object(quality_service, 'comprehensive_quality_analysis', return_value=quality_report):
            result = quality_service.update_test_plans_with_quality_insights([original_plan])
            
            assert len(result) == 1
            updated_plan = result[0]
            
            assert updated_plan.source_file == "file1.py"
            assert updated_plan.weak_mutation_spots == [weak_spot]
            assert updated_plan.quality_score_target == 80.0  # Moderate improvement
            assert updated_plan.mutation_score_target == 85.0  # Moderate improvement
    
    def test_update_test_plans_with_quality_insights_with_poor_existing_tests(self, quality_service):
        """Test updating test plans with quality insights for plans with poor existing tests."""
        original_plan = TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=["test_file1.py"],
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        quality_report = QualityAndMutationReport(
            source_file="file1.py",
            test_files=["test_file1.py"],
            quality_report=TestQualityReport(
                test_file="test_file1.py",
                overall_score=40.0,  # Poor score
                improvement_suggestions=[],
                priority_fixes=[]
            ),
            mutation_score=MutationScore(
                source_file="file1.py",
                test_files=["test_file1.py"],
                total_mutants=10,
                killed_mutants=3,
                surviving_mutants=7,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=30.0,  # Poor score
                mutation_score_effective=30.0,
                weak_spots=[]
            ),
            overall_rating="poor",
            confidence_level=0.3
        )
        
        with patch.object(quality_service, 'comprehensive_quality_analysis', return_value=quality_report):
            result = quality_service.update_test_plans_with_quality_insights([original_plan])
            
            assert len(result) == 1
            updated_plan = result[0]
            
            assert updated_plan.quality_score_target == 60.0  # Lower target for poor tests
            assert updated_plan.mutation_score_target == 65.0  # Lower target for poor tests
    
    def test_update_test_plans_with_quality_insights_with_excellent_existing_tests(self, quality_service):
        """Test updating test plans with quality insights for plans with excellent existing tests."""
        original_plan = TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=["test_file1.py"],
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        quality_report = QualityAndMutationReport(
            source_file="file1.py",
            test_files=["test_file1.py"],
            quality_report=TestQualityReport(
                test_file="test_file1.py",
                overall_score=85.0,  # Excellent score
                improvement_suggestions=[],
                priority_fixes=[]
            ),
            mutation_score=MutationScore(
                source_file="file1.py",
                test_files=["test_file1.py"],
                total_mutants=10,
                killed_mutants=9,
                surviving_mutants=1,
                timeout_mutants=0,
                error_mutants=0,
                mutation_score=90.0,  # Excellent score
                mutation_score_effective=90.0,
                weak_spots=[]
            ),
            overall_rating="excellent",
            confidence_level=0.9
        )
        
        with patch.object(quality_service, 'comprehensive_quality_analysis', return_value=quality_report):
            result = quality_service.update_test_plans_with_quality_insights([original_plan])
            
            assert len(result) == 1
            updated_plan = result[0]
            
            assert updated_plan.quality_score_target == 90.0  # High target for excellent tests
            assert updated_plan.mutation_score_target == 95.0  # High target for excellent tests
    
    def test_update_test_plans_with_quality_insights_without_existing_tests(self, quality_service):
        """Test updating test plans with quality insights for plans without existing tests."""
        original_plan = TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=[],  # No existing tests
            missing_test_elements=[],
            quality_score_target=80.0,
            mutation_score_target=85.0
        )
        
        result = quality_service.update_test_plans_with_quality_insights([original_plan])
        
        assert len(result) == 1
        updated_plan = result[0]
        
        # Should remain unchanged
        assert updated_plan.source_file == "file1.py"
        assert updated_plan.existing_test_files == []
        assert updated_plan.quality_score_target == 80.0
        assert updated_plan.mutation_score_target == 85.0
    
    def test_update_test_plans_with_quality_insights_with_empty_plans(self, quality_service):
        """Test updating test plans with quality insights for empty plans list."""
        result = quality_service.update_test_plans_with_quality_insights([])
        assert result == []
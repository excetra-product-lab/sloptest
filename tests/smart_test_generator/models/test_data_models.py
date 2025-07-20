import pytest
from smart_test_generator.models.data_models import (
    QualityDimension,
    QualityScore,
    TestQualityReport,
    MutationType,
    Mutant,
    MutationResult,
    MutationScore,
    WeakSpot,
    QualityAndMutationReport
)


class TestTestQualityReport:
    """Test TestQualityReport class."""
    
    def test_get_score_returns_existing_dimension_score(self):
        """Test that get_score returns the score for an existing dimension."""
        # Arrange
        dimension = QualityDimension.EDGE_CASE_COVERAGE
        quality_score = QualityScore(dimension=dimension, score=85.5)
        report = TestQualityReport(
            test_file="test_example.py",
            overall_score=80.0,
            dimension_scores={dimension: quality_score}
        )
        
        # Act
        result = report.get_score(dimension)
        
        # Assert
        assert result == 85.5
    
    def test_get_score_returns_zero_for_missing_dimension(self):
        """Test that get_score returns 0.0 for a dimension that doesn't exist."""
        # Arrange
        report = TestQualityReport(
            test_file="test_example.py",
            overall_score=80.0,
            dimension_scores={}
        )
        
        # Act
        result = report.get_score(QualityDimension.ASSERTION_STRENGTH)
        
        # Assert
        assert result == 0.0
    
    def test_get_score_handles_all_quality_dimensions(self):
        """Test that get_score works correctly for all quality dimensions."""
        # Arrange
        scores = {
            QualityDimension.EDGE_CASE_COVERAGE: QualityScore(QualityDimension.EDGE_CASE_COVERAGE, 90.0),
            QualityDimension.ASSERTION_STRENGTH: QualityScore(QualityDimension.ASSERTION_STRENGTH, 75.0),
            QualityDimension.MAINTAINABILITY: QualityScore(QualityDimension.MAINTAINABILITY, 85.0),
            QualityDimension.BUG_DETECTION_POTENTIAL: QualityScore(QualityDimension.BUG_DETECTION_POTENTIAL, 80.0),
            QualityDimension.READABILITY: QualityScore(QualityDimension.READABILITY, 95.0),
            QualityDimension.INDEPENDENCE: QualityScore(QualityDimension.INDEPENDENCE, 70.0)
        }
        report = TestQualityReport(
            test_file="test_comprehensive.py",
            overall_score=82.5,
            dimension_scores=scores
        )
        
        # Act & Assert
        assert report.get_score(QualityDimension.EDGE_CASE_COVERAGE) == 90.0
        assert report.get_score(QualityDimension.ASSERTION_STRENGTH) == 75.0
        assert report.get_score(QualityDimension.MAINTAINABILITY) == 85.0
        assert report.get_score(QualityDimension.BUG_DETECTION_POTENTIAL) == 80.0
        assert report.get_score(QualityDimension.READABILITY) == 95.0
        assert report.get_score(QualityDimension.INDEPENDENCE) == 70.0


class TestMutationScore:
    """Test MutationScore class."""
    
    def test_get_surviving_mutants_returns_unkilled_mutants(self):
        """Test that get_surviving_mutants returns only mutants that weren't killed."""
        # Arrange
        mutant1 = Mutant(
            id="m1",
            mutation_type=MutationType.ARITHMETIC_OPERATOR,
            original_code="x + y",
            mutated_code="x - y",
            line_number=10,
            column_start=5,
            column_end=10,
            description="Changed + to -"
        )
        mutant2 = Mutant(
            id="m2",
            mutation_type=MutationType.COMPARISON_OPERATOR,
            original_code="x == y",
            mutated_code="x != y",
            line_number=15,
            column_start=8,
            column_end=13,
            description="Changed == to !="
        )
        mutant3 = Mutant(
            id="m3",
            mutation_type=MutationType.LOGICAL_OPERATOR,
            original_code="a and b",
            mutated_code="a or b",
            line_number=20,
            column_start=2,
            column_end=9,
            description="Changed and to or"
        )
        
        results = [
            MutationResult(mutant=mutant1, killed=False, execution_time=0.1),
            MutationResult(mutant=mutant2, killed=True, execution_time=0.2),
            MutationResult(mutant=mutant3, killed=False, execution_time=0.15)
        ]
        
        mutation_score = MutationScore(
            source_file="example.py",
            test_files=["test_example.py"],
            total_mutants=3,
            killed_mutants=1,
            surviving_mutants=2,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=33.33,
            mutation_score_effective=33.33,
            mutant_results=results
        )
        
        # Act
        surviving = mutation_score.get_surviving_mutants()
        
        # Assert
        assert len(surviving) == 2
        assert mutant1 in surviving
        assert mutant3 in surviving
        assert mutant2 not in surviving
    
    def test_get_surviving_mutants_returns_empty_when_all_killed(self):
        """Test that get_surviving_mutants returns empty list when all mutants are killed."""
        # Arrange
        mutant = Mutant(
            id="m1",
            mutation_type=MutationType.CONSTANT_VALUE,
            original_code="return True",
            mutated_code="return False",
            line_number=5,
            column_start=7,
            column_end=11,
            description="Changed True to False"
        )
        
        results = [
            MutationResult(mutant=mutant, killed=True, execution_time=0.1)
        ]
        
        mutation_score = MutationScore(
            source_file="perfect.py",
            test_files=["test_perfect.py"],
            total_mutants=1,
            killed_mutants=1,
            surviving_mutants=0,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=100.0,
            mutation_score_effective=100.0,
            mutant_results=results
        )
        
        # Act
        surviving = mutation_score.get_surviving_mutants()
        
        # Assert
        assert surviving == []
    
    def test_get_weak_spots_by_severity_filters_correctly(self):
        """Test that get_weak_spots_by_severity returns spots with matching severity."""
        # Arrange
        mutant1 = Mutant(
            id="m1",
            mutation_type=MutationType.BOUNDARY_VALUE,
            original_code="range(10)",
            mutated_code="range(9)",
            line_number=12,
            column_start=6,
            column_end=15,
            description="Off-by-one error"
        )
        mutant2 = Mutant(
            id="m2",
            mutation_type=MutationType.CONDITIONAL,
            original_code="if x > 0:",
            mutated_code="if x >= 0:",
            line_number=18,
            column_start=3,
            column_end=10,
            description="Changed > to >="
        )
        
        weak_spots = [
            WeakSpot(
                location="example.py:12:6",
                mutation_type=MutationType.BOUNDARY_VALUE,
                surviving_mutants=[mutant1],
                description="Boundary condition not tested",
                severity="high"
            ),
            WeakSpot(
                location="example.py:18:3",
                mutation_type=MutationType.CONDITIONAL,
                surviving_mutants=[mutant2],
                description="Edge case in conditional",
                severity="medium"
            ),
            WeakSpot(
                location="example.py:25:1",
                mutation_type=MutationType.ARITHMETIC_OPERATOR,
                surviving_mutants=[],
                description="Arithmetic operation weakness",
                severity="high"
            )
        ]
        
        mutation_score = MutationScore(
            source_file="example.py",
            test_files=["test_example.py"],
            total_mutants=2,
            killed_mutants=0,
            surviving_mutants=2,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=0.0,
            mutation_score_effective=0.0,
            weak_spots=weak_spots
        )
        
        # Act
        high_severity = mutation_score.get_weak_spots_by_severity("high")
        medium_severity = mutation_score.get_weak_spots_by_severity("medium")
        low_severity = mutation_score.get_weak_spots_by_severity("low")
        
        # Assert
        assert len(high_severity) == 2
        assert len(medium_severity) == 1
        assert len(low_severity) == 0
        assert weak_spots[0] in high_severity
        assert weak_spots[2] in high_severity
        assert weak_spots[1] in medium_severity
    
    def test_get_weak_spots_by_severity_returns_empty_for_nonexistent_severity(self):
        """Test that get_weak_spots_by_severity returns empty list for non-existent severity."""
        # Arrange
        mutation_score = MutationScore(
            source_file="example.py",
            test_files=["test_example.py"],
            total_mutants=0,
            killed_mutants=0,
            surviving_mutants=0,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=0.0,
            mutation_score_effective=0.0,
            weak_spots=[]
        )
        
        # Act
        result = mutation_score.get_weak_spots_by_severity("critical")
        
        # Assert
        assert result == []


class TestQualityAndMutationReport:
    """Test QualityAndMutationReport class."""
    
    def test_get_combined_score_calculates_weighted_average(self):
        """Test that get_combined_score calculates correct weighted average."""
        # Arrange
        quality_report = TestQualityReport(
            test_file="test_example.py",
            overall_score=80.0
        )
        
        mutation_score = MutationScore(
            source_file="example.py",
            test_files=["test_example.py"],
            total_mutants=10,
            killed_mutants=7,
            surviving_mutants=3,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=70.0,
            mutation_score_effective=70.0
        )
        
        report = QualityAndMutationReport(
            source_file="example.py",
            test_files=["test_example.py"],
            quality_report=quality_report,
            mutation_score=mutation_score,
            overall_rating="good",
            confidence_level=0.75
        )
        
        # Act
        combined_score = report.get_combined_score()
        
        # Assert
        # Expected: (80.0 * 0.4) + (70.0 * 0.6) = 32.0 + 42.0 = 74.0
        assert combined_score == 74.0
    
    def test_get_combined_score_handles_perfect_scores(self):
        """Test that get_combined_score handles perfect quality and mutation scores."""
        # Arrange
        quality_report = TestQualityReport(
            test_file="test_perfect.py",
            overall_score=100.0
        )
        
        mutation_score = MutationScore(
            source_file="perfect.py",
            test_files=["test_perfect.py"],
            total_mutants=20,
            killed_mutants=20,
            surviving_mutants=0,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=100.0,
            mutation_score_effective=100.0
        )
        
        report = QualityAndMutationReport(
            source_file="perfect.py",
            test_files=["test_perfect.py"],
            quality_report=quality_report,
            mutation_score=mutation_score,
            overall_rating="excellent",
            confidence_level=1.0
        )
        
        # Act
        combined_score = report.get_combined_score()
        
        # Assert
        # Expected: (100.0 * 0.4) + (100.0 * 0.6) = 40.0 + 60.0 = 100.0
        assert combined_score == 100.0
    
    def test_get_combined_score_handles_zero_scores(self):
        """Test that get_combined_score handles zero quality and mutation scores."""
        # Arrange
        quality_report = TestQualityReport(
            test_file="test_poor.py",
            overall_score=0.0
        )
        
        mutation_score = MutationScore(
            source_file="poor.py",
            test_files=["test_poor.py"],
            total_mutants=15,
            killed_mutants=0,
            surviving_mutants=15,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=0.0,
            mutation_score_effective=0.0
        )
        
        report = QualityAndMutationReport(
            source_file="poor.py",
            test_files=["test_poor.py"],
            quality_report=quality_report,
            mutation_score=mutation_score,
            overall_rating="poor",
            confidence_level=0.0
        )
        
        # Act
        combined_score = report.get_combined_score()
        
        # Assert
        # Expected: (0.0 * 0.4) + (0.0 * 0.6) = 0.0 + 0.0 = 0.0
        assert combined_score == 0.0
    
    def test_get_combined_score_uses_correct_weights(self):
        """Test that get_combined_score uses the correct quality and mutation weights."""
        # Arrange
        quality_report = TestQualityReport(
            test_file="test_weighted.py",
            overall_score=60.0
        )
        
        mutation_score = MutationScore(
            source_file="weighted.py",
            test_files=["test_weighted.py"],
            total_mutants=8,
            killed_mutants=6,
            surviving_mutants=2,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=75.0,
            mutation_score_effective=75.0
        )
        
        report = QualityAndMutationReport(
            source_file="weighted.py",
            test_files=["test_weighted.py"],
            quality_report=quality_report,
            mutation_score=mutation_score,
            overall_rating="fair",
            confidence_level=0.65
        )
        
        # Act
        combined_score = report.get_combined_score()
        
        # Assert
        # Expected: (60.0 * 0.4) + (75.0 * 0.6) = 24.0 + 45.0 = 69.0
        # This verifies that mutation score has higher weight (0.6) than quality (0.4)
        assert combined_score == 69.0
        
        # Verify the weights favor mutation score
        quality_contribution = 60.0 * 0.4  # 24.0
        mutation_contribution = 75.0 * 0.6  # 45.0
        assert mutation_contribution > quality_contribution
    
    def test_get_combined_score_handles_decimal_precision(self):
        """Test that get_combined_score handles decimal precision correctly."""
        # Arrange
        quality_report = TestQualityReport(
            test_file="test_decimal.py",
            overall_score=77.77
        )
        
        mutation_score = MutationScore(
            source_file="decimal.py",
            test_files=["test_decimal.py"],
            total_mutants=9,
            killed_mutants=7,
            surviving_mutants=2,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=83.33,
            mutation_score_effective=83.33
        )
        
        report = QualityAndMutationReport(
            source_file="decimal.py",
            test_files=["test_decimal.py"],
            quality_report=quality_report,
            mutation_score=mutation_score,
            overall_rating="good",
            confidence_level=0.8
        )
        
        # Act
        combined_score = report.get_combined_score()
        
        # Assert
        # Expected: (77.77 * 0.4) + (83.33 * 0.6) = 31.108 + 49.998 = 81.106
        expected_score = 77.77 * 0.4 + 83.33 * 0.6
        assert abs(combined_score - expected_score) < 0.001

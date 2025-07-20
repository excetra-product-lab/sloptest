"""Quality analysis service integrating test quality scoring and mutation testing."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from smart_test_generator.models.data_models import (
    TestQualityReport, MutationScore, QualityAndMutationReport,
    TestGenerationPlan, WeakSpot
)
from smart_test_generator.analysis.quality_analyzer import TestQualityEngine
from smart_test_generator.analysis.mutation_engine import MutationTestingEngine
from smart_test_generator.analysis.python_analyzers import (
    get_python_quality_analyzers, get_python_mutation_operators
)
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback
from .base_service import BaseService

logger = logging.getLogger(__name__)


class QualityAnalysisService(BaseService):
    """Service for comprehensive test quality analysis and mutation testing."""
    
    def __init__(self, project_root: Path, config: Config, feedback: Optional[UserFeedback] = None):
        super().__init__(project_root, config, feedback)
        
        # Initialize quality analysis engine
        self.quality_engine = TestQualityEngine(
            custom_analyzers=get_python_quality_analyzers()
        )
        
        # Initialize mutation testing engine
        mutation_timeout = self.config.get('quality.mutation_timeout', 30)
        self.mutation_engine = MutationTestingEngine(
            operators=get_python_mutation_operators(),
            timeout=mutation_timeout
        )
        
        # Configuration
        self.max_mutants_per_file = self.config.get('quality.max_mutants_per_file', 50)
        self.quality_threshold = self.config.get('quality.minimum_quality_score', 75.0)
        self.mutation_threshold = self.config.get('quality.minimum_mutation_score', 80.0)
        self.enable_mutation_testing = self.config.get('quality.enable_mutation_testing', True)
    
    def analyze_test_quality(self, test_file: str, source_file: str = "") -> TestQualityReport:
        """Analyze the quality of a test file."""
        try:
            self._log_info(f"Analyzing test quality for {test_file}")
            return self.quality_engine.analyze_test_quality(test_file, source_file)
        except Exception as e:
            self._log_error(f"Failed to analyze test quality for {test_file}: {e}")
            return self._create_empty_quality_report(test_file)
    
    def run_mutation_testing(self, source_file: str, test_files: List[str]) -> Optional[MutationScore]:
        """Run mutation testing for a source file against its test files."""
        if not self.enable_mutation_testing:
            self._log_info("Mutation testing disabled in configuration")
            return None
        
        try:
            self._log_info(f"Running mutation testing for {source_file}")
            return self.mutation_engine.run_mutation_testing(
                source_file, test_files, max_mutants=self.max_mutants_per_file
            )
        except Exception as e:
            self._log_error(f"Failed to run mutation testing for {source_file}: {e}")
            return None
    
    def comprehensive_quality_analysis(self, source_file: str, test_files: List[str]) -> QualityAndMutationReport:
        """Run comprehensive quality analysis including both quality scoring and mutation testing."""
        self._log_info(f"Running comprehensive quality analysis for {source_file}")
        
        # Analyze test quality for all test files
        combined_quality_report = self._analyze_combined_test_quality(test_files, source_file)
        
        # Run mutation testing
        mutation_score = None
        if self.enable_mutation_testing and test_files:
            mutation_score = self.run_mutation_testing(source_file, test_files)
        
        # Create empty mutation score if not available
        if mutation_score is None:
            mutation_score = self._create_empty_mutation_score(source_file, test_files)
        
        # Generate combined report
        return self._create_combined_report(source_file, test_files, combined_quality_report, mutation_score)
    
    def analyze_test_plans_quality(self, test_plans: List[TestGenerationPlan]) -> Dict[str, QualityAndMutationReport]:
        """Analyze quality for multiple test generation plans."""
        quality_reports = {}
        
        for plan in test_plans:
            if plan.existing_test_files:
                quality_report = self.comprehensive_quality_analysis(
                    plan.source_file, plan.existing_test_files
                )
                quality_reports[plan.source_file] = quality_report
        
        return quality_reports
    
    def identify_quality_gaps(self, quality_reports: Dict[str, QualityAndMutationReport]) -> Dict[str, List[str]]:
        """Identify quality gaps across all analyzed files."""
        gaps = {}
        
        for source_file, report in quality_reports.items():
            file_gaps = []
            
            # Check overall quality score
            if report.quality_report.overall_score < self.quality_threshold:
                file_gaps.append(f"Quality score ({report.quality_report.overall_score:.1f}) below threshold ({self.quality_threshold})")
            
            # Check mutation score
            if report.mutation_score.mutation_score < self.mutation_threshold:
                file_gaps.append(f"Mutation score ({report.mutation_score.mutation_score:.1f}%) below threshold ({self.mutation_threshold}%)")
            
            # Add priority fixes
            file_gaps.extend(report.quality_report.priority_fixes)
            
            # Add critical weak spots
            critical_weak_spots = report.mutation_score.get_weak_spots_by_severity("critical")
            for weak_spot in critical_weak_spots:
                file_gaps.append(f"Critical weak spot: {weak_spot.description}")
            
            if file_gaps:
                gaps[source_file] = file_gaps
        
        return gaps
    
    def generate_quality_improvement_suggestions(self, report: QualityAndMutationReport) -> List[str]:
        """Generate prioritized suggestions for improving test quality."""
        suggestions = []
        
        # High priority suggestions from quality analysis
        suggestions.extend(report.quality_report.priority_fixes[:3])
        
        # High priority suggestions from mutation testing
        high_priority_weak_spots = report.mutation_score.get_weak_spots_by_severity("high")
        for weak_spot in high_priority_weak_spots[:2]:  # Top 2 high priority
            suggestions.extend(weak_spot.suggested_tests[:2])  # Top 2 suggestions each
        
        # Critical weak spots always get priority
        critical_weak_spots = report.mutation_score.get_weak_spots_by_severity("critical")
        for weak_spot in critical_weak_spots:
            suggestions.extend(weak_spot.suggested_tests[:1])  # Top suggestion each
        
        # Additional quality improvements
        if report.quality_report.overall_score < 60:
            suggestions.append("Focus on basic test structure and assertions before advanced techniques")
        
        if report.mutation_score.mutation_score < 50:
            suggestions.append("Add more comprehensive test scenarios to catch basic bugs")
        
        return list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order
    
    def update_test_plans_with_quality_insights(self, test_plans: List[TestGenerationPlan]) -> List[TestGenerationPlan]:
        """Update test generation plans with quality analysis insights."""
        updated_plans = []
        
        for plan in test_plans:
            updated_plan = plan
            
            if plan.existing_test_files:
                # Analyze existing test quality
                quality_report = self.comprehensive_quality_analysis(plan.source_file, plan.existing_test_files)
                
                # Update plan with quality insights
                updated_plan.weak_mutation_spots = quality_report.mutation_score.weak_spots
                
                # Adjust quality targets based on current state
                if quality_report.quality_report.overall_score < 50:
                    updated_plan.quality_score_target = 60.0  # Lower target for very poor tests
                elif quality_report.quality_report.overall_score < 75:
                    updated_plan.quality_score_target = 80.0  # Moderate improvement
                else:
                    updated_plan.quality_score_target = 90.0  # High standard for good tests
                
                if quality_report.mutation_score.mutation_score < 50:
                    updated_plan.mutation_score_target = 65.0  # Lower target for very poor tests
                elif quality_report.mutation_score.mutation_score < 80:
                    updated_plan.mutation_score_target = 85.0  # Moderate improvement
                else:
                    updated_plan.mutation_score_target = 95.0  # High standard for good tests
            
            updated_plans.append(updated_plan)
        
        return updated_plans
    
    def _analyze_combined_test_quality(self, test_files: List[str], source_file: str) -> TestQualityReport:
        """Analyze quality of multiple test files combined."""
        if not test_files:
            return self._create_empty_quality_report("no_test_files")
        
        if len(test_files) == 1:
            return self.analyze_test_quality(test_files[0], source_file)
        
        # For multiple test files, combine the analysis
        all_reports = []
        for test_file in test_files:
            report = self.analyze_test_quality(test_file, source_file)
            all_reports.append(report)
        
        # Create combined report
        return self._merge_quality_reports(all_reports, test_files)
    
    def _merge_quality_reports(self, reports: List[TestQualityReport], test_files: List[str]) -> TestQualityReport:
        """Merge multiple quality reports into one."""
        if not reports:
            return self._create_empty_quality_report("no_reports")
        
        if len(reports) == 1:
            return reports[0]
        
        # Calculate weighted average scores
        total_score = sum(r.overall_score for r in reports)
        avg_score = total_score / len(reports)
        
        # Combine all suggestions
        all_suggestions = []
        all_priority_fixes = []
        all_edge_cases_found = []
        all_edge_cases_missing = []
        all_weak_assertions = []
        all_maintainability_issues = []
        
        for report in reports:
            all_suggestions.extend(report.improvement_suggestions)
            all_priority_fixes.extend(report.priority_fixes)
            all_edge_cases_found.extend(report.edge_cases_found)
            all_edge_cases_missing.extend(report.edge_cases_missing)
            all_weak_assertions.extend(report.weak_assertions)
            all_maintainability_issues.extend(report.maintainability_issues)
        
        # Remove duplicates
        all_suggestions = list(dict.fromkeys(all_suggestions))
        all_priority_fixes = list(dict.fromkeys(all_priority_fixes))
        
        return TestQualityReport(
            test_file=f"combined_tests({len(test_files)})",
            overall_score=avg_score,
            edge_cases_found=list(set(all_edge_cases_found)),
            edge_cases_missing=list(set(all_edge_cases_missing)),
            weak_assertions=all_weak_assertions,
            maintainability_issues=list(set(all_maintainability_issues)),
            improvement_suggestions=all_suggestions,
            priority_fixes=all_priority_fixes
        )
    
    def _create_combined_report(self, source_file: str, test_files: List[str], 
                               quality_report: TestQualityReport, 
                               mutation_score: MutationScore) -> QualityAndMutationReport:
        """Create a combined quality and mutation report."""
        
        # Calculate overall rating
        combined_score = quality_report.overall_score * 0.4 + mutation_score.mutation_score * 0.6
        
        if combined_score >= 90:
            overall_rating = "excellent"
        elif combined_score >= 75:
            overall_rating = "good"
        elif combined_score >= 60:
            overall_rating = "fair"
        else:
            overall_rating = "poor"
        
        # Calculate confidence level
        confidence_level = min(combined_score / 100.0, 1.0)
        
        # Generate recommended actions
        recommended_actions = self.generate_quality_improvement_suggestions(
            QualityAndMutationReport(
                source_file=source_file,
                test_files=test_files,
                quality_report=quality_report,
                mutation_score=mutation_score,
                overall_rating=overall_rating,
                confidence_level=confidence_level
            )
        )
        
        # Prioritize fixes
        high_priority_fixes = []
        medium_priority_fixes = []
        low_priority_fixes = []
        
        # Quality-based fixes
        if quality_report.overall_score < 50:
            high_priority_fixes.extend([
                {"type": "quality", "description": fix, "impact": "high"}
                for fix in quality_report.priority_fixes[:2]
            ])
        elif quality_report.overall_score < 75:
            medium_priority_fixes.extend([
                {"type": "quality", "description": fix, "impact": "medium"}
                for fix in quality_report.priority_fixes[:3]
            ])
        
        # Mutation-based fixes
        critical_weak_spots = mutation_score.get_weak_spots_by_severity("critical")
        high_weak_spots = mutation_score.get_weak_spots_by_severity("high")
        
        for weak_spot in critical_weak_spots:
            high_priority_fixes.append({
                "type": "mutation",
                "description": weak_spot.description,
                "suggestions": weak_spot.suggested_tests,
                "impact": "critical"
            })
        
        for weak_spot in high_weak_spots[:2]:  # Limit to top 2
            medium_priority_fixes.append({
                "type": "mutation", 
                "description": weak_spot.description,
                "suggestions": weak_spot.suggested_tests,
                "impact": "high"
            })
        
        return QualityAndMutationReport(
            source_file=source_file,
            test_files=test_files,
            quality_report=quality_report,
            mutation_score=mutation_score,
            overall_rating=overall_rating,
            confidence_level=confidence_level,
            recommended_actions=recommended_actions,
            high_priority_fixes=high_priority_fixes,
            medium_priority_fixes=medium_priority_fixes,
            low_priority_fixes=low_priority_fixes
        )
    
    def _create_empty_quality_report(self, test_file: str) -> TestQualityReport:
        """Create an empty quality report for error cases."""
        return TestQualityReport(
            test_file=test_file,
            overall_score=0.0,
            improvement_suggestions=["Unable to analyze test quality"],
            priority_fixes=["Check file exists and is readable"]
        )
    
    def _create_empty_mutation_score(self, source_file: str, test_files: List[str]) -> MutationScore:
        """Create an empty mutation score for error cases."""
        return MutationScore(
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
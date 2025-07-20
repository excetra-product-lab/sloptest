"""Coverage analysis service."""

from pathlib import Path
from typing import Dict, List, Optional

from smart_test_generator.models.data_models import TestCoverage
from smart_test_generator.analysis.coverage_analyzer import CoverageAnalyzer
from smart_test_generator.tracking.state_tracker import TestGenerationTracker
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback
from .base_service import BaseService


class CoverageService(BaseService):
    """Service for handling coverage analysis operations."""
    
    def __init__(self, project_root: Path, config: Config, feedback: Optional[UserFeedback] = None):
        super().__init__(project_root, config, feedback)
        self.coverage_analyzer = CoverageAnalyzer(project_root, config)
        self.tracker = TestGenerationTracker()
    
    def analyze_coverage(self, files: List[str]) -> Dict[str, TestCoverage]:
        """Run coverage analysis on specified files."""
        with self.feedback.status_spinner("Running coverage analysis"):
            try:
                coverage_data = self.coverage_analyzer.run_coverage_analysis(files)
                
                if not coverage_data:
                    self.feedback.warning("No coverage data available. Make sure you have tests and pytest-cov installed.")
                    return {}
                
                # Update coverage history for each file
                for filepath, coverage in coverage_data.items():
                    self.tracker.update_coverage(filepath, coverage.line_coverage)
                
                self.tracker.save_state()
                
                # Show compact coverage summary
                if coverage_data:
                    avg_coverage = sum(cov.line_coverage for cov in coverage_data.values()) / len(coverage_data)
                    self.feedback.success(f"Coverage analysis complete: {avg_coverage:.1f}% average")
                
                return coverage_data
                
            except Exception as e:
                self.feedback.error(f"Coverage analysis failed: {e}")
                raise
    
    def generate_coverage_report(self, coverage_data: Dict[str, TestCoverage]) -> str:
        """Generate a coverage report summary."""
        if not coverage_data:
            return "No coverage data available."
        
        total_coverage = sum(cov.line_coverage for cov in coverage_data.values()) / len(coverage_data)
        
        report_lines = [
            f"Overall coverage: {total_coverage:.1f}%",
            ""
        ]
        
        for filepath, coverage in coverage_data.items():
            rel_path = str(Path(filepath).relative_to(self.project_root))
            report_lines.append(f"{rel_path}: {coverage.line_coverage:.1f}%")
        
        return "\n".join(report_lines)
    
    def get_coverage_history(self) -> Dict[str, List[float]]:
        """Get coverage history for all tracked files."""
        return self.tracker.state.coverage_history.copy() 
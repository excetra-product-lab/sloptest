"""Analysis service for code analysis operations."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage, QualityAndMutationReport
from smart_test_generator.utils.parser import PythonCodebaseParser
from smart_test_generator.generation.test_generator import IncrementalTestGenerator
from smart_test_generator.tracking.state_tracker import TestGenerationTracker
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback
from .base_service import BaseService
from .coverage_service import CoverageService
from .quality_service import QualityAnalysisService


class AnalysisService(BaseService):
    """"""
    
    def __init__(self, target_dir: Path, project_root: Path, config: Config, feedback: Optional[UserFeedback] = None):
        super().__init__(project_root, config, feedback)
        self.target_dir = target_dir  # Directory to scan
        # Initialize parser with target directory for scanning
        self.parser = PythonCodebaseParser(str(target_dir), config)
        self.test_generator = IncrementalTestGenerator(project_root, config)
        self.tracker = TestGenerationTracker()
        self.coverage_service = CoverageService(project_root, config, feedback)
        self.quality_service = QualityAnalysisService(project_root, config, feedback)
    
    def find_python_files(self) -> List[str]:
        """Find all Python files in the project."""
        return self.parser.find_python_files()
    
    def analyze_files_for_generation(self, files: List[str], force: bool = False) -> Tuple[List[str], Dict[str, str], Dict[str, TestCoverage]]:
        """Analyze which files need test generation."""
        self._log_info("Analyzing files for test generation...")
        
        # Get coverage data
        coverage_data = self.coverage_service.analyze_coverage(files)
        
        # Determine which files need test generation
        files_to_process = []
        reasons = {}
        
        for filepath in files:
            coverage = coverage_data.get(filepath)
            should_generate, reason = self.tracker.should_generate_tests(
                filepath, coverage, self.config, force
            )
            
            # Add debugging information for better visibility
            if should_generate:
                # Check if tests actually exist for this file to provide better reason
                existing_tests = self.test_generator.test_mapper.find_test_files(filepath)
                if existing_tests:
                    enhanced_reason = f"{reason} (Note: Found {len(existing_tests)} existing test file(s): {[str(Path(t).name) for t in existing_tests]})"
                    self._log_info(f"Including {filepath} for generation: {enhanced_reason}")
                    reasons[filepath] = enhanced_reason
                else:
                    self._log_info(f"Including {filepath} for generation: {reason}")
                    reasons[filepath] = reason
                files_to_process.append(filepath)
            else:
                self._log_info(f"Skipping {filepath}: {reason}")
        
        if not files_to_process:
            self._log_info("No files need test generation - all files have adequate coverage or existing tests")
        else:
            self._log_info(f"Will process {len(files_to_process)} files for test generation")
        
        return files_to_process, reasons, coverage_data
    
    def create_test_plans(self, files: List[str], coverage_data: Dict[str, TestCoverage]) -> List[TestGenerationPlan]:
        """Create test generation plans for the specified files."""
        test_plans = []
        
        with self.feedback.status_spinner("Creating test generation plans"):
            for filepath in files:
                coverage = coverage_data.get(filepath)
                plan = self.test_generator.generate_test_plan(filepath, coverage)
                if plan.elements_to_test:
                    test_plans.append(plan)
        
        # Enhance test plans with quality insights
        if test_plans:
            with self.feedback.status_spinner("Enhancing plans with quality analysis"):
                test_plans = self.quality_service.update_test_plans_with_quality_insights(test_plans)
        
        # Show compact summary
        if test_plans:
            total_elements = sum(len(plan.elements_to_test) for plan in test_plans)
            self.feedback.success(f"Created {len(test_plans)} test plans covering {total_elements} code elements")
        else:
            self.feedback.info("No test plans needed - all code elements are already tested")
        
        return test_plans
    
    def analyze_test_quality(self, test_plans: List[TestGenerationPlan]) -> Dict[str, QualityAndMutationReport]:
        """Analyze test quality for existing test files."""
        if not test_plans:
            return {}
        
        # Count files that have existing tests
        files_with_tests = [plan for plan in test_plans if plan.existing_test_files]
        
        if not files_with_tests:
            self.feedback.info("No existing test files found for quality analysis")
            return {}
        
        with self.feedback.status_spinner(f"Analyzing quality of {len(files_with_tests)} test suites"):
            quality_reports = self.quality_service.analyze_test_plans_quality(test_plans)
        
        if quality_reports:
            avg_quality = sum(report.quality_report.overall_score for report in quality_reports.values()) / len(quality_reports)
            avg_mutation = sum(report.mutation_score.mutation_score for report in quality_reports.values()) / len(quality_reports)
            self.feedback.success(f"Quality analysis complete: {avg_quality:.1f}% avg quality, {avg_mutation:.1f}% avg mutation score")
        
        return quality_reports
    
    def generate_quality_gaps_report(self, quality_reports: Dict[str, QualityAndMutationReport]) -> str:
        """Generate a report of quality gaps and improvement suggestions."""
        gaps = self.quality_service.identify_quality_gaps(quality_reports)
        
        if not gaps:
            return "✅ No significant quality gaps found in existing tests!"
        
        report_lines = [
            "Test Quality Analysis",
            "=" * 50,
            ""
        ]
        
        for source_file, issues in gaps.items():
            report_lines.append(f"📁 {source_file}")
            for issue in issues:
                report_lines.append(f"  ⚠️  {issue}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def generate_analysis_report(self, all_files: List[str], files_to_process: List[str], 
                               test_plans: List[TestGenerationPlan], reasons: Dict[str, str]) -> str:
        """Generate a comprehensive analysis report."""
        report_lines = [
            "Analysis Report",
            "=" * 50,
            f"Total Python files: {len(all_files)}",
            f"Files needing test generation: {len(files_to_process)}",
            ""
        ]
        
        if files_to_process:
            report_lines.append("Files to process:")
            for filepath in files_to_process:
                rel_path = os.path.relpath(filepath, self.project_root)
                report_lines.append(f"  {rel_path}: {reasons[filepath]}")
            
            report_lines.append("")
            total_untested = sum(len(plan.elements_to_test) for plan in test_plans)
            report_lines.append(f"Total elements needing tests: {total_untested}")
        else:
            report_lines.append("All files have adequate test coverage!")
        
        return "\n".join(report_lines)
    
    def get_generation_status(self) -> str:
        """Get the current generation status and history."""
        if not self.tracker.state.generation_log:
            return "No test generation history found."
        
        status_lines = [
            "Test Generation Status",
            "=" * 50,
            ""
        ]
        
        # Show last 10 entries
        for entry in self.tracker.state.generation_log[-10:]:
            status_lines.extend([
                f"{entry['timestamp']}:",
                f"  File: {entry['filepath']}",
                f"  Reason: {entry['reason']}",
                f"  Elements generated: {entry['elements_generated']}",
                f"  Coverage: {entry['coverage_before']:.1f}% → {entry['coverage_after']:.1f}%",
                ""
            ])
        
        return "\n".join(status_lines) 

    def debug_test_generation_state(self) -> str:
        """Debug the current state of test generation tracking."""
        state_summary = self.tracker.get_state_summary()
        
        debug_info = [
            "=== Test Generation State Debug ===",
            f"State timestamp: {state_summary['timestamp']}",
            f"Files with recorded tests: {state_summary['files_with_tests']}",
            f"Total tested elements: {state_summary['total_tested_elements']}",
            f"Files with coverage history: {state_summary['files_with_coverage_history']}",
            f"Generation log entries: {state_summary['generation_log_entries']}",
            "",
            "Files with recorded tests:"
        ]
        
        for filepath in state_summary['tested_files']:
            tested_elements = self.tracker.state.tested_elements.get(filepath, [])
            debug_info.append(f"  {filepath}: {len(tested_elements)} elements")
            
            # Check if test files actually exist for this source file
            existing_tests = self.test_generator.test_mapper.find_test_files(filepath)
            if existing_tests:
                debug_info.append(f"    -> Found test files: {[str(Path(t).name) for t in existing_tests]}")
            else:
                debug_info.append(f"    -> WARNING: No test files found!")
        
        return "\n".join(debug_info)

    def sync_state_with_existing_tests(self) -> str:
        """Sync state tracker with existing tests to avoid regenerating."""
        synced_files = 0
        total_elements_synced = 0
        
        # Get all Python files that might have tests
        all_files = self.find_python_files()
        
        for filepath in all_files:
            # Find existing test files for this source file
            existing_tests = self.test_generator.test_mapper.find_test_files(filepath)
            
            if existing_tests:
                # Extract all tested elements from existing test files
                all_tested = []
                
                # Use the correct TestMapper method
                tested_elements = self.test_generator.test_mapper.analyze_test_completeness(filepath, existing_tests)
                
                # Flatten the tested elements into a single list
                all_tested.extend(tested_elements['functions'])
                all_tested.extend(tested_elements['classes'])
                all_tested.extend(tested_elements['methods'])
                
                if all_tested:
                    # Mark elements as tested in state tracker
                    self.tracker.force_mark_as_tested(
                        filepath, 
                        all_tested, 
                        f"Synced with existing tests: {[str(Path(t).name) for t in existing_tests]}"
                    )
                    synced_files += 1
                    total_elements_synced += len(all_tested)
                    
                    # Only show per-file details in verbose mode
                    if self.feedback.verbose:
                        rel_path = os.path.relpath(filepath, self.project_root)
                        if len(rel_path) > 70:
                            rel_path = "..." + rel_path[-67:]
                        self.feedback.debug(f"Synced {len(all_tested)} elements for {rel_path}")
        
        # Show compact summary instead of verbose per-file logging
        if synced_files > 0:
            summary_msg = f"Sync complete: {total_elements_synced} elements across {synced_files} files"
            self.feedback.success(summary_msg)
            return summary_msg
        else:
            no_sync_msg = "No existing tests found to sync"
            self.feedback.info(no_sync_msg)
            return no_sync_msg

    def reset_generation_state(self) -> str:
        """Reset the test generation state (use when state becomes corrupted)."""
        self._log_info("Resetting test generation state...")
        self.tracker.reset_state()
        return "Test generation state has been reset" 
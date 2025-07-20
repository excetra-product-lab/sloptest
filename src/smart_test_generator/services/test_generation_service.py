"""Test generation service."""

from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import os

from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage
from smart_test_generator.generation.llm_clients import LLMClient
from smart_test_generator.generation.incremental_generator import IncrementalLLMClient
from smart_test_generator.utils.writer import TestFileWriter
from smart_test_generator.tracking.state_tracker import TestGenerationTracker
from smart_test_generator.reporting.reporter import TestGenerationReporter
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback, ProgressTracker
from smart_test_generator.exceptions import TestGenerationError
from .base_service import BaseService


class TestFileResult(NamedTuple):
    """Result of writing a single test file."""
    source_path: str
    success: bool
    error_message: Optional[str] = None


class TestGenerationService(BaseService):
    """Service for handling test generation operations."""
    
    def __init__(self, project_root: Path, config: Config, feedback: Optional[UserFeedback] = None):
        super().__init__(project_root, config, feedback)
        self.tracker = TestGenerationTracker()
        self.writer = TestFileWriter(str(project_root), config)
        self.reporter = TestGenerationReporter(project_root)
    
    def generate_tests(self, llm_client: LLMClient, test_plans: List[TestGenerationPlan], 
                      directory_structure: str, batch_size: int = 10, 
                      generation_reasons: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate tests using hybrid approach: LLM batching + streaming writes."""
        if not test_plans:
            self.feedback.info("No test plans provided for generation")
            return {}
        
        progress = ProgressTracker(self.feedback)
        progress.set_total_steps(3, "Test Generation")
        
        # Track results for summary
        written_files = []
        failed_files = []
        
        try:
            # Initialize incremental generation client
            progress.step("Initializing test generation")
            incremental_client = IncrementalLLMClient(llm_client, self.config)
            
            # Process test plans in batches
            progress.step("Generating and writing tests")
            
            total_batches = (len(test_plans) + batch_size - 1) // batch_size
            
            # Show compact batch overview
            self.feedback.sophisticated_progress("Processing test plans", f"{len(test_plans)} plans in {total_batches} batches")
            
            for i in range(0, len(test_plans), batch_size):
                batch = test_plans[i:i + batch_size]
                batch_num = i // batch_size + 1
                batch_end = min(i + batch_size, len(test_plans))
                
                # Compact batch progress
                with self.feedback.status_spinner(f"Batch {batch_num}/{total_batches}: plans {i+1}-{batch_end}"):
                    try:
                        # Collect source files for validation
                        batch_source_files = [plan.source_file for plan in batch]
                        
                        # Generate tests for this batch (keep LLM batching for efficiency)
                        batch_tests = incremental_client.generate_contextual_tests(
                            batch, directory_structure, batch_source_files, str(self.project_root))
                        
                        # Stream write each test file immediately
                        batch_results = self._write_batch_immediately(batch, batch_tests, generation_reasons or {})
                        
                        # Track results
                        for result in batch_results:
                            if result.success:
                                written_files.append(result.source_path)
                            else:
                                failed_files.append(result.source_path)
                    
                        # Compact success indicator
                        successful_count = len([r for r in batch_results if r.success])
                        failed_count = len([r for r in batch_results if not r.success])
                        
                        if failed_count == 0:
                            self.feedback.success(f"✓ Batch {batch_num}: {successful_count} tests generated")
                        else:
                            self.feedback.warning(f"⚠ Batch {batch_num}: {successful_count} success, {failed_count} failed")
                        
                    except Exception as e:
                        self.feedback.error(f"✗ Batch {batch_num} failed: {e}")
                        # Track all files in this batch as failed
                        for plan in batch:
                            failed_files.append(plan.source_file)
                        # Continue with next batch instead of failing completely
                        continue
            
            if not written_files:
                raise TestGenerationError(
                    "Failed to generate any tests",
                    suggestion="Check your API credentials and network connection. Try again with fewer files."
                )
            
            # Record final activity summary
            progress.step("Recording generation summary")
            self._record_batch_summary(written_files, failed_files)
            
            # Create summary for return compatibility
            summary_dict = {file: f"Generated and written successfully" for file in written_files}
            
            # Show final results table
            self._show_generation_results_summary(written_files, failed_files)
            
            progress.complete(f"Generated tests for {len(written_files)} files")
            return summary_dict
            
        except Exception as e:
            progress.error(f"Test generation failed: {e}")
            
            # Save any partial progress
            if written_files:
                self.feedback.info(f"Partial progress: {len(written_files)} files completed")
                self._record_batch_summary(written_files, failed_files)
            
            raise

    def generate_tests_streaming(self, llm_client: LLMClient, test_plans: List[TestGenerationPlan], 
                                directory_structure: str, 
                                generation_reasons: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate tests with true file-by-file streaming - write each test as soon as it's ready."""
        if not test_plans:
            self.feedback.info("No test plans provided for generation")
            return {}
        
        progress = ProgressTracker(self.feedback)
        progress.set_total_steps(2, "Streaming Test Generation")
        
        # Track results for summary
        written_files = []
        failed_files = []
        
        try:
            # Initialize incremental generation client
            progress.step("Initializing streaming test generation")
            incremental_client = IncrementalLLMClient(llm_client, self.config)
            
            # Process each test plan individually
            progress.step("Generating and writing tests individually")
            
            total_files = len(test_plans)
            self.feedback.sophisticated_progress("Streaming generation", f"{total_files} files, one at a time")
            
            for i, plan in enumerate(test_plans, 1):
                rel_path = os.path.relpath(plan.source_file, self.project_root)
                if len(rel_path) > 50:
                    rel_path = "..." + rel_path[-47:]
                
                with self.feedback.status_spinner(f"File {i}/{total_files}: {rel_path}"):
                    try:
                        # Generate test for this single file
                        test_content = incremental_client.generate_single_file_test(
                            plan, directory_structure, [plan.source_file], str(self.project_root))
                        
                        if test_content:
                            # Write test file immediately
                            result = self._write_single_test_file(plan, test_content, generation_reasons or {})
                            
                            if result.success:
                                written_files.append(result.source_path)
                                self.feedback.success(f"✓ {rel_path}")
                            else:
                                failed_files.append(result.source_path)
                                self.feedback.error(f"✗ {rel_path}: {result.error_message}")
                        else:
                            failed_files.append(plan.source_file)
                            self.feedback.warning(f"⚠ {rel_path}: No test content generated")
                            
                    except Exception as e:
                        failed_files.append(plan.source_file)
                        self.feedback.error(f"✗ {rel_path}: {e}")
                        continue
            
            if not written_files:
                raise TestGenerationError(
                    "Failed to generate any tests",
                    suggestion="Check your API credentials and network connection. Try again with fewer files."
                )
            
            # Record final activity summary
            self._record_batch_summary(written_files, failed_files)
            
            # Show final results table
            self._show_generation_results_summary(written_files, failed_files)
            
            # Create summary for return compatibility
            summary_dict = {file: f"Generated and written successfully" for file in written_files}
            
            progress.complete(f"Streaming complete: {len(written_files)} tests generated")
            return summary_dict
            
        except Exception as e:
            progress.error(f"Streaming generation failed: {e}")
            
            # Save any partial progress
            if written_files:
                self.feedback.info(f"Partial progress: {len(written_files)} files completed")
                self._record_batch_summary(written_files, failed_files)
            
            raise
    
    def _write_batch_immediately(self, batch_plans: List[TestGenerationPlan], 
                               batch_tests: Dict[str, str], 
                               generation_reasons: Dict[str, str]) -> List[TestFileResult]:
        """Write test files immediately and update tracking incrementally."""
        results = []
        batch_feedback = []  # Collect feedback messages for display after batch
        
        for plan in batch_plans:
            source_path = plan.source_file
            
            if source_path in batch_tests:
                test_content = batch_tests[source_path]
                
                # Write the test file immediately
                write_result = self._write_single_test_file(source_path, test_content)
                results.append(write_result)
                
                # Update tracking incrementally for successful writes
                if write_result.success:
                    self._update_tracking_incremental(plan, generation_reasons.get(source_path, "Generated"))
                    batch_feedback.append(("success", f"✓ Test written: {source_path}"))
                else:
                    batch_feedback.append(("error", f"✗ Failed to write: {source_path} - {write_result.error_message}"))
            else:
                # Plan was processed but no test content was generated
                result = TestFileResult(source_path, False, "No test content generated")
                results.append(result)
                batch_feedback.append(("warning", f"⚠ No test generated for: {source_path}"))
        
        # Display all feedback for this batch at once (when progress tracker is not interfering)
        self._display_batch_feedback(batch_feedback)
        
        return results
    
    def _display_batch_feedback(self, batch_feedback: List[tuple]):
        """Display batch feedback messages without progress tracker interference."""
        if not batch_feedback:
            return
            
        # Add a small pause to ensure progress tracker updates are flushed
        import time
        time.sleep(0.1)
        
        # Use console print directly to avoid Rich Progress interference
        for msg_type, message in batch_feedback:
            if msg_type == "success":
                # Use direct console print instead of feedback.success to avoid interference
                self.feedback.console.print(f"[bold green]✓[/bold green] {message}")
            elif msg_type == "error":
                self.feedback.console.print(f"[bold red]✗[/bold red] {message}")
            elif msg_type == "warning":
                self.feedback.console.print(f"[bold yellow]⚠[/bold yellow] {message}")
        
        # Add a visual separator between batches if there were messages
        if len(batch_feedback) > 0:
            self.feedback.console.print("[dim]────────────────────────────────────────────[/dim]")
    
    def _write_single_test_file(self, source_path: str, test_content: str) -> TestFileResult:
        """Write a single test file and return the result."""
        try:
            self.writer.write_test_file(source_path, test_content)
            return TestFileResult(source_path, True)
        except Exception as e:
            return TestFileResult(source_path, False, str(e))
    
    def _update_tracking_incremental(self, plan: TestGenerationPlan, reason: str):
        """Update tracking state for a single successful test generation."""
        elements_generated = [e.name for e in plan.elements_to_test]
        coverage_before = plan.coverage_before.line_coverage if plan.coverage_before else 0
        
        self.tracker.record_generation(
            plan.source_file,
            elements_generated,
            coverage_before,
            plan.estimated_coverage_after,
            reason
        )
        
        # Save state after each file for fault tolerance
        self.tracker.save_state()
    
    def _record_batch_summary(self, written_files: List[str], failed_files: List[str]):
        """Record a summary of the batch processing results."""
        total_files = len(written_files) + len(failed_files)
        self._log_info(f"Generation completed: {len(written_files)}/{total_files} files successful")
        
        if failed_files:
            self._log_warning(f"Failed files: {failed_files}")
    
    def _write_test_files(self, tests_dict: Dict[str, str]):
        """Write generated test files to disk."""
        self._log_info(f"Writing {len(tests_dict)} test files to disk...")
        
        written_count = 0
        for source_path, test_content in tests_dict.items():
            try:
                self._log_info(f"Writing test file for: {source_path}")
                self.writer.write_test_file(source_path, test_content)
                written_count += 1
                self._log_info(f"Successfully wrote test file for: {source_path}")
            except Exception as e:
                self._log_error(f"Failed to write test file for {source_path}: {e}")
                # Continue writing other files instead of failing completely
                continue
        
        self._log_info(f"Successfully wrote {written_count} out of {len(tests_dict)} test files")
    
    def _record_generation_activity(self, test_plans: List[TestGenerationPlan], 
                                  generated_tests: Dict[str, str], 
                                  generation_reasons: Dict[str, str]):
        """Record test generation activity in the state tracker."""
        for plan in test_plans:
            if plan.source_file in generated_tests:
                elements_generated = [e.name for e in plan.elements_to_test]
                coverage_before = plan.coverage_before.line_coverage if plan.coverage_before else 0
                reason = generation_reasons.get(plan.source_file, "Unknown reason")
                
                self.tracker.record_generation(
                    plan.source_file,
                    elements_generated,
                    coverage_before,
                    plan.estimated_coverage_after,
                    reason
                )
        
        self.tracker.save_state()
    
    def measure_coverage_improvement(self, files_processed: List[str], 
                                   old_coverage_data: Dict[str, TestCoverage],
                                   coverage_service) -> Dict[str, float]:
        """Measure coverage improvement after test generation."""
        try:
            # Run coverage analysis again to see improvement
            self.feedback.progress("Measuring coverage improvement")
            new_coverage_data = coverage_service.analyze_coverage(files_processed)
            
            # Calculate improvements
            coverage_before = sum(
                old_coverage_data.get(f, type('TestCoverage', (), {'line_coverage': 0})()).line_coverage
                for f in files_processed
            ) / len(files_processed) if files_processed else 0
            
            coverage_after = sum(
                new_coverage_data.get(f, type('TestCoverage', (), {'line_coverage': 0})()).line_coverage
                for f in files_processed
            ) / len(files_processed) if files_processed else 0
            
            return {
                'before': coverage_before,
                'after': coverage_after,
                'improvement': coverage_after - coverage_before
            }
            
        except Exception as e:
            self._log_warning(f"Could not measure coverage improvement: {e}")
            return {'before': 0, 'after': 0, 'improvement': 0}
    
    def generate_final_report(self, generated_tests: Dict[str, str], 
                            coverage_improvement: Dict[str, float]) -> str:
        """Generate a final report of the test generation process."""
        results = {
            "files_processed": len(generated_tests),
            "tests_generated": len(generated_tests),
            "coverage_before": coverage_improvement.get('before', 0),
            "coverage_after": coverage_improvement.get('after', 0),
            "details": list(generated_tests.keys())
        }
        
        return self.reporter.generate_report(results) 

    def _show_generation_results_summary(self, written_files: List[str], failed_files: List[str]):
        """Show a compact visual summary of generation results."""
        total_files = len(written_files) + len(failed_files)
        success_rate = (len(written_files) / total_files * 100) if total_files > 0 else 0
        
        # Results summary table
        summary_items = [
            ("success", "Generated", f"{len(written_files)} test files"),
            ("error" if failed_files else "success", "Failed", f"{len(failed_files)} files"),
            ("info", "Success Rate", f"{success_rate:.1f}%"),
        ]
        
        self.feedback.status_table("📈 Generation Results", summary_items)
        
        # Show failed files if any (and not too many)
        if failed_files and len(failed_files) <= 5:
            self.feedback.subsection_header(f"Failed Files ({len(failed_files)})")
            for filepath in failed_files:
                rel_path = os.path.relpath(filepath, self.project_root)
                if len(rel_path) > 70:
                    rel_path = "..." + rel_path[-67:]
                self.feedback.error(f"✗ {rel_path}")
        elif failed_files and len(failed_files) > 5:
            self.feedback.warning(f"⚠ {len(failed_files)} files failed (use -v to see details)")
            
            # In verbose mode, show all failed files
            if self.feedback.verbose:
                self.feedback.subsection_header(f"Failed Files ({len(failed_files)})")
                for filepath in failed_files:
                    rel_path = os.path.relpath(filepath, self.project_root)
                    if len(rel_path) > 70:
                        rel_path = "..." + rel_path[-67:]
                    self.feedback.error(f"✗ {rel_path}")
        
        # Show success message
        if written_files:
            if len(written_files) <= 5:
                self.feedback.subsection_header(f"Generated Tests ({len(written_files)})")
                for filepath in written_files:
                    rel_path = os.path.relpath(filepath, self.project_root)
                    if len(rel_path) > 70:
                        rel_path = "..." + rel_path[-67:]
                    self.feedback.success(f"✓ {rel_path}")
            else:
                self.feedback.success(f"✓ Successfully generated {len(written_files)} test files") 
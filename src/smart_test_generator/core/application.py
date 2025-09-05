"""Main application orchestrator."""

from pathlib import Path
from typing import Dict, List, Optional

from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.utils.cost_manager import CostManager
from smart_test_generator.services import TestGenerationService, CoverageService, AnalysisService
from smart_test_generator.core.llm_factory import LLMClientFactory
from smart_test_generator.exceptions import SmartTestGeneratorError


class SmartTestGeneratorApp:
    """Main application orchestrator that coordinates between services."""
    
    def __init__(self, target_dir: Path, project_root: Path, config: Config, feedback: Optional[UserFeedback] = None):
        self.target_dir = target_dir  # Directory to scan for files
        self.project_root = project_root  # Project root for configuration
        self.config = config
        self.feedback = feedback or UserFeedback()
        
        # Initialize services with target directory for scanning but project root for config
        self.analysis_service = AnalysisService(target_dir, project_root, config, feedback)
        self.coverage_service = CoverageService(project_root, config, feedback)
        self.test_generation_service = TestGenerationService(project_root, config, feedback)
    
    def _ensure_state_synced(self) -> None:
        """Ensure state tracker is synced with existing tests before any operation."""
        try:
            # Don't show verbose sync messages - let the spinner handle the display
            sync_result = self.analysis_service.sync_state_with_existing_tests()
            self.feedback.debug(sync_result)
        except Exception as e:
            self.feedback.warning(f"State sync failed, continuing with current state: {e}")

    def run_analysis_mode(self, force: bool = False) -> str:
        """Run analysis mode to show what would be done."""
        try:
            # Always sync state with existing tests first
            self._ensure_state_synced()
            
            # Find all Python files
            all_files = self.analysis_service.find_python_files()
            
            # Analyze which files need test generation
            files_to_process, reasons, coverage_data = self.analysis_service.analyze_files_for_generation(
                all_files, force
            )
            
            # Create test plans
            test_plans = self.analysis_service.create_test_plans(files_to_process, coverage_data)
            
            # Show beautiful test plans display
            if test_plans:
                self.feedback.test_plans_display(test_plans, self.project_root)
            else:
                self.feedback.success("All files have adequate test coverage! No generation needed.")
            
            # Analyze test quality for existing tests
            quality_reports = self.analysis_service.analyze_test_quality(test_plans)
            
            # Display beautiful quality analysis results
            if quality_reports:
                self.feedback.quality_analysis_display(quality_reports, self.project_root)
            
            # Return a summary for compatibility
            total_elements = sum(len(plan.elements_to_test) for plan in test_plans)
            return f"Analysis complete: {len(test_plans)} files need tests, {total_elements} elements to test"
            
        except Exception as e:
            self.feedback.error(f"Analysis failed: {e}")
            raise
    
    def run_coverage_mode(self) -> str:
        """Run coverage analysis mode."""
        try:
            # Always sync state with existing tests first
            self._ensure_state_synced()
            
            # Find all Python files
            all_files = self.analysis_service.find_python_files()
            
            # Run coverage analysis
            coverage_data = self.coverage_service.analyze_coverage(all_files)
            
            # Generate and return coverage report
            return self.coverage_service.generate_coverage_report(coverage_data)
            
        except Exception as e:
            self.feedback.error(f"Coverage analysis failed: {e}")
            raise
    
    def run_status_mode(self) -> str:
        """Run status mode to show generation history."""
        try:
            # Always sync state with existing tests first
            self._ensure_state_synced()
            
            return self.analysis_service.get_generation_status()
        except Exception as e:
            self.feedback.error(f"Status retrieval failed: {e}")
            raise
    
    def run_generate_mode(self, llm_credentials: Dict, batch_size: int = 1, 
                         force: bool = False, dry_run: bool = False, 
                         streaming: bool = False) -> str:
        """Run test generation mode."""
        try:
            # Always sync state with existing tests first
            self._ensure_state_synced()
            
            # Create cost manager for tracking LLM usage
            cost_manager = CostManager(self.config)
            
            # Create LLM client
            self.feedback.info("Initializing LLM client")
            llm_client = LLMClientFactory.create_client(
                claude_api_key=llm_credentials.get('claude_api_key'),
                claude_model=llm_credentials.get('claude_model', 'claude-3-5-sonnet-20241022'),
                claude_extended_thinking=llm_credentials.get('claude_extended_thinking', False),
                claude_thinking_budget=llm_credentials.get('claude_thinking_budget'),
                azure_endpoint=llm_credentials.get('azure_endpoint'),
                azure_api_key=llm_credentials.get('azure_api_key'),
                azure_deployment=llm_credentials.get('azure_deployment'),
                bedrock_role_arn=llm_credentials.get('bedrock_role_arn'),
                bedrock_inference_profile=llm_credentials.get('bedrock_inference_profile'),
                bedrock_region=llm_credentials.get('bedrock_region'),
                feedback=self.feedback,
                cost_manager=cost_manager,
                config=self.config
            )
            
            # Find all Python files
            self.feedback.info("Scanning for Python files")
            all_files = self.analysis_service.find_python_files()
            
            # Analyze which files need test generation
            files_to_process, generation_reasons, coverage_data = self.analysis_service.analyze_files_for_generation(
                all_files, force
            )
            
            if not files_to_process:
                self.feedback.success("All files have adequate test coverage! No generation needed.")
                return "No generation needed - all files have adequate coverage."
            
            # Create test plans
            test_plans = self.analysis_service.create_test_plans(files_to_process, coverage_data)
            
            if not test_plans:
                self.feedback.success("No untested elements found. All code appears to be tested!")
                return "No untested elements found."
            
            # Show what will be processed with beautiful display
            self.feedback.test_plans_display(test_plans, self.project_root)
            
            if dry_run:
                return f"Dry run - would generate tests for {len(test_plans)} files."
            
            # Generate directory structure for context
            directory_structure = self.analysis_service.parser.generate_directory_structure()
            
            # Generate tests
            if streaming:
                self.feedback.info("Using streaming mode - tests will be written as each file is ready")
                generated_tests = self.test_generation_service.generate_tests_streaming(
                    llm_client, test_plans, directory_structure, generation_reasons
                )
            else:
                generated_tests = self.test_generation_service.generate_tests(
                    llm_client, test_plans, directory_structure, batch_size, generation_reasons
                )
            
            # Measure coverage improvement
            coverage_improvement = self.test_generation_service.measure_coverage_improvement(
                files_to_process, coverage_data, self.coverage_service
            )
            
            # Run quality analysis on newly generated tests (if enabled)
            quality_reports = {}
            if self.config.get('quality.enable_quality_analysis', True):
                self.feedback.info("Running quality analysis on generated tests...")
                
                # Re-create test plans to include newly generated test files
                updated_files_to_process, _, updated_coverage_data = self.analysis_service.analyze_files_for_generation(
                    files_to_process, force=True  # Force re-analysis to pick up new test files
                )
                updated_test_plans = self.analysis_service.create_test_plans(updated_files_to_process, updated_coverage_data)
                
                # Run quality analysis including mutation testing
                quality_reports = self.analysis_service.analyze_test_quality(updated_test_plans)
                
                # Display beautiful quality analysis results
                if quality_reports:
                    self.feedback.quality_analysis_display(quality_reports, self.project_root)
                else:
                    self.feedback.info("Quality analysis skipped - no existing test files found for analysis")
            else:
                self.feedback.info("Quality analysis disabled in configuration")
            
            # Generate final report
            final_report = self.test_generation_service.generate_final_report(
                generated_tests, coverage_improvement
            )
            
            # Show summary
            summary_data = {
                "Files processed": len(test_plans),
                "Tests generated": len(generated_tests),
                "Coverage before": f"{coverage_improvement.get('before', 0):.1f}%",
                "Coverage after": f"{coverage_improvement.get('after', 0):.1f}%",
                "Improvement": f"+{coverage_improvement.get('improvement', 0):.1f}%"
            }
            
            # Add quality metrics to summary if available
            if quality_reports:
                avg_quality = sum(report.quality_report.overall_score for report in quality_reports.values()) / len(quality_reports)
                avg_mutation = sum(report.mutation_score.mutation_score for report in quality_reports.values()) / len(quality_reports)
                summary_data["Average quality score"] = f"{avg_quality:.1f}%"
                summary_data["Average mutation score"] = f"{avg_mutation:.1f}%"
            
            self.feedback.summary("Test Generation & Quality Summary", summary_data)
            
            # Show cost statistics after generation
            self._show_cost_statistics(cost_manager)
            
            return final_report
            
        except Exception as e:
            self.feedback.error(f"Test generation failed: {e}")
            raise
    
    def run_debug_state_mode(self) -> str:
        """Run debug state mode to show current state information."""
        try:
            return self.analysis_service.debug_test_generation_state()
        except Exception as e:
            self.feedback.error(f"State debugging failed: {e}")
            raise
    
    def run_sync_state_mode(self) -> str:
        """Run sync state mode to sync state with existing tests."""
        try:
            return self.analysis_service.sync_state_with_existing_tests()
        except Exception as e:
            self.feedback.error(f"State sync failed: {e}")
            raise
    
    def run_reset_state_mode(self) -> str:
        """Run reset state mode to reset test generation state."""
        try:
            return self.analysis_service.reset_generation_state()
        except Exception as e:
            self.feedback.error(f"State reset failed: {e}")
            raise
    
    def initialize_config(self, config_file: str = ".testgen.yml") -> str:
        """Initialize a sample configuration file."""
        try:
            self.config.create_sample_config(config_file)
            return f"Sample configuration created at {config_file}"
        except Exception as e:
            self.feedback.error(f"Failed to create configuration: {e}")
            raise
    
    def _show_cost_statistics(self, cost_manager: CostManager) -> None:
        """Display cost statistics after test generation."""
        try:
            # Get usage summary for this session (current day)
            usage_summary = cost_manager.get_usage_summary(days=1)
            
            if usage_summary.get('requests', 0) > 0:
                # Show cost statistics with nice formatting
                cost_info = {
                    "Session Cost": f"${usage_summary['total_cost']:.4f}",
                    "API Requests": str(usage_summary['requests']),
                    "Total Tokens": f"{usage_summary['total_tokens']:,}",
                    "Avg Cost/Request": f"${usage_summary['average_cost_per_request']:.4f}"
                }
                
                self.feedback.summary_panel("💰 Cost Statistics", cost_info, "yellow")
                
                # Show optimization tips if cost is significant
                if usage_summary['total_cost'] > 0.50:  # More than 50 cents
                    self.feedback.info("💡 Cost Optimization Tips:")
                    self.feedback.info("• Use smaller batch sizes: --batch-size 5")
                    self.feedback.info("• Try streaming mode: --streaming") 
                    self.feedback.info("• Use cheaper models for simple files: --claude-model claude-3-5-haiku-20241022")
            else:
                self.feedback.debug("No cost data available for this session")
                
        except Exception as e:
            self.feedback.debug(f"Could not display cost statistics: {e}") 
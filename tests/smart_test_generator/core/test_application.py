import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from smart_test_generator.core.application import SmartTestGeneratorApp
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.exceptions import SmartTestGeneratorError
from smart_test_generator.models.data_models import TestGenerationPlan, TestableElement


class TestSmartTestGeneratorApp:
    """Test suite for SmartTestGeneratorApp class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.create_sample_config = Mock()
        return config
    
    @pytest.fixture
    def mock_feedback(self):
        """Create a mock feedback object."""
        feedback = Mock(spec=UserFeedback)
        feedback.info = Mock()
        feedback.error = Mock()
        feedback.success = Mock()
        feedback.summary = Mock()
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
    @patch('smart_test_generator.core.application.TestGenerationService')
    @patch('smart_test_generator.core.application.CoverageService')
    @patch('smart_test_generator.core.application.AnalysisService')
    def app(self, mock_analysis_service, mock_coverage_service, mock_test_generation_service, 
            target_dir, project_root, mock_config, mock_feedback):
        """Create a SmartTestGeneratorApp instance with mocked services."""
        return SmartTestGeneratorApp(target_dir, project_root, mock_config, mock_feedback)
    
    def test_init_with_all_parameters(self, target_dir, project_root, mock_config, mock_feedback):
        """Test initialization with all parameters provided."""
        with patch('smart_test_generator.core.application.AnalysisService') as mock_analysis, \
             patch('smart_test_generator.core.application.CoverageService') as mock_coverage, \
             patch('smart_test_generator.core.application.TestGenerationService') as mock_test_gen:
            
            app = SmartTestGeneratorApp(target_dir, project_root, mock_config, mock_feedback)
            
            assert app.target_dir == target_dir
            assert app.project_root == project_root
            assert app.config == mock_config
            assert app.feedback == mock_feedback
            
            # Verify services are initialized with correct parameters
            mock_analysis.assert_called_once_with(target_dir, project_root, mock_config, mock_feedback)
            mock_coverage.assert_called_once_with(project_root, mock_config, mock_feedback)
            mock_test_gen.assert_called_once_with(project_root, mock_config, mock_feedback)
    
    def test_init_without_feedback(self, target_dir, project_root, mock_config):
        """Test initialization without feedback parameter creates default UserFeedback."""
        with patch('smart_test_generator.core.application.AnalysisService'), \
             patch('smart_test_generator.core.application.CoverageService'), \
             patch('smart_test_generator.core.application.TestGenerationService'), \
             patch('smart_test_generator.core.application.UserFeedback') as mock_user_feedback:
            
            mock_feedback_instance = Mock()
            mock_user_feedback.return_value = mock_feedback_instance
            
            app = SmartTestGeneratorApp(target_dir, project_root, mock_config)
            
            assert app.feedback == mock_feedback_instance
            mock_user_feedback.assert_called_once_with()
    
    def test_run_analysis_mode_success(self, app):
        """Test successful analysis mode execution."""
        # Setup mock return values
        all_files = [Path("file1.py"), Path("file2.py")]
        files_to_process = [Path("file1.py")]
        reasons = {"file1.py": "No test file found"}
        coverage_data = {"file1.py": 0.5}
        test_plans = [TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=[],
            elements_to_test=[TestableElement("function1", "function", "file1.py", 10, "def function1():")],
            coverage_before=None,
            estimated_coverage_after=80.0
        )]
        quality_reports = [{"file": "test_file1.py", "quality": "good"}]
        analysis_report = "Analysis Report"
        quality_gaps_report = "Quality Gaps Report"
        
        app.analysis_service.find_python_files.return_value = all_files
        app.analysis_service.analyze_files_for_generation.return_value = (files_to_process, reasons, coverage_data)
        app.analysis_service.create_test_plans.return_value = test_plans
        app.analysis_service.analyze_test_quality.return_value = quality_reports
        app.analysis_service.generate_analysis_report.return_value = analysis_report
        app.analysis_service.generate_quality_gaps_report.return_value = quality_gaps_report
        
        result = app.run_analysis_mode(force=True)
        
        assert result == "Analysis complete: 1 files need tests, 1 elements to test"
        app.analysis_service.find_python_files.assert_called_once()
        app.analysis_service.analyze_files_for_generation.assert_called_once_with(all_files, True)
        app.analysis_service.create_test_plans.assert_called_once_with(files_to_process, coverage_data)
        app.analysis_service.analyze_test_quality.assert_called_once_with(test_plans)
        # Note: generate_analysis_report and generate_quality_gaps_report are no longer called
        # in the new implementation - the analysis mode uses direct feedback display instead
    
    def test_run_analysis_mode_with_exception(self, app):
        """Test analysis mode handles exceptions properly."""
        app.analysis_service.find_python_files.side_effect = Exception("File system error")
        
        with pytest.raises(Exception, match="File system error"):
            app.run_analysis_mode()
        
        app.feedback.error.assert_called_once_with("Analysis failed: File system error")
    
    def test_run_coverage_mode_success(self, app):
        """Test successful coverage mode execution."""
        all_files = [Path("file1.py"), Path("file2.py")]
        coverage_data = {"file1.py": 0.8, "file2.py": 0.6}
        coverage_report = "Coverage Report"
        
        app.analysis_service.find_python_files.return_value = all_files
        app.coverage_service.analyze_coverage.return_value = coverage_data
        app.coverage_service.generate_coverage_report.return_value = coverage_report
        
        result = app.run_coverage_mode()
        
        assert result == "Coverage Report"
        app.analysis_service.find_python_files.assert_called_once()
        app.coverage_service.analyze_coverage.assert_called_once_with(all_files)
        app.coverage_service.generate_coverage_report.assert_called_once_with(coverage_data)
    
    def test_run_coverage_mode_with_exception(self, app):
        """Test coverage mode handles exceptions properly."""
        app.analysis_service.find_python_files.side_effect = Exception("Coverage error")
        
        with pytest.raises(Exception, match="Coverage error"):
            app.run_coverage_mode()
        
        app.feedback.error.assert_called_once_with("Coverage analysis failed: Coverage error")
    
    def test_run_status_mode_success(self, app):
        """Test successful status mode execution."""
        status_report = "Generation Status Report"
        app.analysis_service.get_generation_status.return_value = status_report
        
        result = app.run_status_mode()
        
        assert result == "Generation Status Report"
        app.analysis_service.get_generation_status.assert_called_once()
    
    def test_run_status_mode_with_exception(self, app):
        """Test status mode handles exceptions properly."""
        app.analysis_service.get_generation_status.side_effect = Exception("Status error")
        
        with pytest.raises(Exception, match="Status error"):
            app.run_status_mode()
        
        app.feedback.error.assert_called_once_with("Status retrieval failed: Status error")
    
    @patch('smart_test_generator.core.application.LLMClientFactory')
    def test_run_generate_mode_success_full_flow(self, mock_llm_factory, app):
        """Test successful test generation mode with full flow."""
        # Setup
        llm_credentials = {
            'claude_api_key': 'test-key',
            'claude_model': 'claude-3-5-sonnet-20241022'
        }
        mock_llm_client = Mock()
        mock_llm_factory.create_client.return_value = mock_llm_client
        
        all_files = [Path("file1.py")]
        files_to_process = [Path("file1.py")]
        generation_reasons = {"file1.py": "No test file found"}
        coverage_data = {"file1.py": 0.5}
        test_plans = [TestGenerationPlan(
            source_file="file1.py",
            existing_test_files=[],
            elements_to_test=[TestableElement("function1", "function", "file1.py", 10, "def function1():")],
            coverage_before=None,
            estimated_coverage_after=80.0
        )]
        analysis_report = "Analysis Report"
        directory_structure = "Directory Structure"
        generated_tests = [{"file": "test_file1.py", "content": "test content"}]
        coverage_improvement = {"before": 50.0, "after": 80.0, "improvement": 30.0}
        final_report = "Final Report"
        
        app.analysis_service.find_python_files.return_value = all_files
        app.analysis_service.analyze_files_for_generation.return_value = (files_to_process, generation_reasons, coverage_data)
        app.analysis_service.create_test_plans.return_value = test_plans
        app.analysis_service.generate_analysis_report.return_value = analysis_report
        app.analysis_service.parser.generate_directory_structure.return_value = directory_structure
        app.test_generation_service.generate_tests.return_value = generated_tests
        app.test_generation_service.measure_coverage_improvement.return_value = coverage_improvement
        app.test_generation_service.generate_final_report.return_value = final_report
        
        result = app.run_generate_mode(llm_credentials, batch_size=5, force=True, dry_run=False)
        
        assert result == "Final Report"
        
        # Verify LLM client creation
        mock_llm_factory.create_client.assert_called_once_with(
            claude_api_key='test-key',
            claude_model='claude-3-5-sonnet-20241022',
            claude_extended_thinking=False,
            claude_thinking_budget=None,
            azure_endpoint=None,
            azure_api_key=None,
            azure_deployment=None,
            bedrock_role_arn=None,
            bedrock_inference_profile=None,
            bedrock_region=None,
            feedback=app.feedback,
            cost_manager=mock_llm_factory.create_client.call_args[1]['cost_manager'],
            config=app.config
        )
        
        # Verify test generation flow
        app.test_generation_service.generate_tests.assert_called_once_with(
            mock_llm_client, test_plans, directory_structure, 5, generation_reasons
        )
        
        # Verify feedback calls
        app.feedback.summary.assert_called_once_with("Test Generation Summary", {
            "Files processed": 1,
            "Tests generated": 1,
            "Coverage before": "50.0%",
            "Coverage after": "80.0%",
            "Improvement": "+30.0%"
        })
    
    @patch('smart_test_generator.core.application.LLMClientFactory')
    def test_run_generate_mode_no_files_to_process(self, mock_llm_factory, app):
        """Test generation mode when no files need processing."""
        llm_credentials = {'claude_api_key': 'test-key'}
        mock_llm_client = Mock()
        mock_llm_factory.create_client.return_value = mock_llm_client
        
        all_files = [Path("file1.py")]
        files_to_process = []  # No files to process
        generation_reasons = {}
        coverage_data = {}
        
        app.analysis_service.find_python_files.return_value = all_files
        app.analysis_service.analyze_files_for_generation.return_value = (files_to_process, generation_reasons, coverage_data)
        
        result = app.run_generate_mode(llm_credentials)
        
        assert result == "No generation needed - all files have adequate coverage."
        app.feedback.success.assert_called_once_with("All files have adequate test coverage! No generation needed.")
    
    @patch('smart_test_generator.core.application.LLMClientFactory')
    def test_run_generate_mode_no_test_plans(self, mock_llm_factory, app):
        """Test generation mode when no test plans are created."""
        llm_credentials = {'claude_api_key': 'test-key'}
        mock_llm_client = Mock()
        mock_llm_factory.create_client.return_value = mock_llm_client
        
        all_files = [Path("file1.py")]
        files_to_process = [Path("file1.py")]
        generation_reasons = {"file1.py": "No test file found"}
        coverage_data = {"file1.py": 0.5}
        test_plans = []  # No test plans
        
        app.analysis_service.find_python_files.return_value = all_files
        app.analysis_service.analyze_files_for_generation.return_value = (files_to_process, generation_reasons, coverage_data)
        app.analysis_service.create_test_plans.return_value = test_plans
        
        result = app.run_generate_mode(llm_credentials)
        
        assert result == "No untested elements found."
        app.feedback.success.assert_called_once_with("No untested elements found. All code appears to be tested!")
    
    @patch('smart_test_generator.core.application.LLMClientFactory')
    def test_run_generate_mode_dry_run(self, mock_llm_factory, app):
        """Test generation mode in dry run mode."""
        llm_credentials = {'claude_api_key': 'test-key'}
        mock_llm_client = Mock()
        mock_llm_factory.create_client.return_value = mock_llm_client
        
        all_files = [Path("file1.py")]
        files_to_process = [Path("file1.py")]
        generation_reasons = {"file1.py": "No test file found"}
        coverage_data = {"file1.py": 0.5}
        test_plans = [
            TestGenerationPlan(
                source_file="file1.py",
                existing_test_files=[],
                elements_to_test=[TestableElement("function1", "function", "file1.py", 10, "def function1():")],
                coverage_before=None,
                estimated_coverage_after=80.0
            ),
            TestGenerationPlan(
                source_file="file2.py",
                existing_test_files=[],
                elements_to_test=[TestableElement("function2", "function", "file2.py", 20, "def function2():")],
                coverage_before=None,
                estimated_coverage_after=75.0
            )
        ]
        analysis_report = "Analysis Report"
        
        app.analysis_service.find_python_files.return_value = all_files
        app.analysis_service.analyze_files_for_generation.return_value = (files_to_process, generation_reasons, coverage_data)
        app.analysis_service.create_test_plans.return_value = test_plans
        app.analysis_service.generate_analysis_report.return_value = analysis_report
        
        result = app.run_generate_mode(llm_credentials, dry_run=True)
        
        assert result == "Dry run - would generate tests for 2 files."
        # In dry run mode, analysis report is not displayed, instead we see scanning message
        app.feedback.info.assert_any_call("Scanning for Python files")
    
    @patch('smart_test_generator.core.application.LLMClientFactory')
    def test_run_generate_mode_with_azure_credentials(self, mock_llm_factory, app):
        """Test generation mode with Azure credentials."""
        llm_credentials = {
            'azure_endpoint': 'https://test.openai.azure.com',
            'azure_api_key': 'azure-key',
            'azure_deployment': 'gpt-4'
        }
        mock_llm_client = Mock()
        mock_llm_factory.create_client.return_value = mock_llm_client
        
        # Setup minimal successful flow
        app.analysis_service.find_python_files.return_value = []
        app.analysis_service.analyze_files_for_generation.return_value = ([], {}, {})
        
        result = app.run_generate_mode(llm_credentials)
        
        mock_llm_factory.create_client.assert_called_once_with(
            claude_api_key=None,
            claude_model='claude-3-5-sonnet-20241022',
            claude_extended_thinking=False,
            claude_thinking_budget=None,
            azure_endpoint='https://test.openai.azure.com',
            azure_api_key='azure-key',
            azure_deployment='gpt-4',
            bedrock_role_arn=None,
            bedrock_inference_profile=None,
            bedrock_region=None,
            feedback=app.feedback,
            cost_manager=mock_llm_factory.create_client.call_args[1]['cost_manager'],
            config=app.config
        )
    
    @patch('smart_test_generator.core.application.LLMClientFactory')
    def test_run_generate_mode_with_exception(self, mock_llm_factory, app):
        """Test generation mode handles exceptions properly."""
        llm_credentials = {'claude_api_key': 'test-key'}
        mock_llm_factory.create_client.side_effect = Exception("LLM client error")
        
        with pytest.raises(Exception, match="LLM client error"):
            app.run_generate_mode(llm_credentials)
        
        app.feedback.error.assert_called_once_with("Test generation failed: LLM client error")
    
    def test_initialize_config_success(self, app):
        """Test successful config initialization."""
        config_file = "custom_config.yml"
        
        result = app.initialize_config(config_file)
        
        assert result == f"Sample configuration created at {config_file}"
        app.config.create_sample_config.assert_called_once_with(config_file)
    
    def test_initialize_config_default_filename(self, app):
        """Test config initialization with default filename."""
        result = app.initialize_config()
        
        assert result == "Sample configuration created at .testgen.yml"
        app.config.create_sample_config.assert_called_once_with(".testgen.yml")
    
    def test_initialize_config_with_exception(self, app):
        """Test config initialization handles exceptions properly."""
        app.config.create_sample_config.side_effect = Exception("Config creation error")
        
        with pytest.raises(Exception, match="Config creation error"):
            app.initialize_config()
        
        app.feedback.error.assert_called_once_with("Failed to create configuration: Config creation error")
    
    def test_run_generate_mode_batch_size_parameter(self, app):
        """Test that batch_size parameter is passed correctly."""
        with patch('smart_test_generator.core.application.LLMClientFactory') as mock_llm_factory:
            llm_credentials = {'claude_api_key': 'test-key'}
            mock_llm_client = Mock()
            mock_llm_factory.create_client.return_value = mock_llm_client
            
            # Setup for successful generation
            all_files = [Path("file1.py")]
            files_to_process = [Path("file1.py")]
            generation_reasons = {"file1.py": "No test file found"}
            coverage_data = {"file1.py": 0.5}
            test_plans = [TestGenerationPlan(
                source_file="file1.py",
                existing_test_files=[],
                elements_to_test=[TestableElement("function1", "function", "file1.py", 10, "def function1():")],
                coverage_before=None,
                estimated_coverage_after=80.0
            )]
            
            app.analysis_service.find_python_files.return_value = all_files
            app.analysis_service.analyze_files_for_generation.return_value = (files_to_process, generation_reasons, coverage_data)
            app.analysis_service.create_test_plans.return_value = test_plans
            app.analysis_service.generate_analysis_report.return_value = "Analysis Report"
            app.analysis_service.parser.generate_directory_structure.return_value = "Directory Structure"
            app.test_generation_service.generate_tests.return_value = []
            app.test_generation_service.measure_coverage_improvement.return_value = {"before": 0, "after": 0, "improvement": 0}
            app.test_generation_service.generate_final_report.return_value = "Final Report"
            
            app.run_generate_mode(llm_credentials, batch_size=15)
            
            # Verify batch_size is passed to generate_tests
            app.test_generation_service.generate_tests.assert_called_once_with(
                mock_llm_client, test_plans, "Directory Structure", 15, generation_reasons
            )
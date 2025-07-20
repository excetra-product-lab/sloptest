import pytest
import argparse
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from smart_test_generator.cli import (
    show_welcome_banner,
    setup_argparse,
    validate_system_and_project,
    load_and_validate_config,
    validate_arguments,
    extract_llm_credentials,
    handle_init_config_mode,
    execute_mode_with_status,
    main
)
from smart_test_generator.exceptions import (
    ValidationError,
    ConfigurationError,
    DependencyError,
    ProjectStructureError,
    SmartTestGeneratorError,
    AuthenticationError
)


class TestShowWelcomeBanner:
    """Test show_welcome_banner function."""
    
    def test_displays_welcome_banner_with_features(self):
        """Test that welcome banner displays title and features."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        
        # Act
        show_welcome_banner(mock_feedback)
        
        # Assert
        mock_feedback.section_header.assert_called_once_with("Smart Test Generator")
        mock_feedback.feature_showcase.assert_called_once()
        features = mock_feedback.feature_showcase.call_args[0][0]
        assert "Intelligent test analysis" in features
        assert "Coverage-driven generation" in features
        assert "Quality assessment" in features
        assert "Incremental updates" in features
        mock_feedback.divider.assert_called_once()


class TestSetupArgparse:
    """Test setup_argparse function."""
    
    def test_creates_parser_with_all_arguments(self):
        """Test that parser is created with all expected arguments."""
        # Act
        parser = setup_argparse()
        
        # Assert
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == "Generate unit tests for Python codebase"
        
        # Test that we can parse basic arguments
        args = parser.parse_args(['generate', '--directory', '/test'])
        assert args.mode == 'generate'
        assert args.directory == '/test'
    
    def test_default_values_are_set_correctly(self):
        """Test that default argument values are correct."""
        # Act
        parser = setup_argparse()
        args = parser.parse_args([])
        
        # Assert
        assert args.mode == 'generate'
        assert args.directory == '.'
        assert args.batch_size == 10
        assert args.config == '.testgen.yml'
        assert args.claude_model == 'claude-sonnet-4-20250514'
        assert args.usage_days == 7
        assert args.force is False
        assert args.dry_run is False
        assert args.verbose is False
        assert args.cost_optimize is False
    
    def test_mode_choices_are_valid(self):
        """Test that only valid modes are accepted."""
        # Act
        parser = setup_argparse()
        
        # Assert - valid modes should work
        valid_modes = ['generate', 'analyze', 'coverage', 'status', 'init-config', 'cost']
        for mode in valid_modes:
            args = parser.parse_args([mode])
            assert args.mode == mode
    
    def test_boolean_flags_work_correctly(self):
        """Test that boolean flags can be set."""
        # Act
        parser = setup_argparse()
        args = parser.parse_args(['--force', '--dry-run', '--verbose', '--cost-optimize'])
        
        # Assert
        assert args.force is True
        assert args.dry_run is True
        assert args.verbose is True
        assert args.cost_optimize is True
    
    def test_numeric_arguments_are_parsed_correctly(self):
        """Test that numeric arguments are parsed as integers/floats."""
        # Act
        parser = setup_argparse()
        args = parser.parse_args(['--batch-size', '5', '--max-cost', '10.50', '--usage-days', '14'])
        
        # Assert
        assert args.batch_size == 5
        assert args.max_cost == 10.50
        assert args.usage_days == 14


class TestValidateSystemAndProject:
    """Test validate_system_and_project function."""
    
    @patch('smart_test_generator.cli.SystemValidator')
    @patch('smart_test_generator.cli.Validator')
    def test_successful_validation_returns_paths(self, mock_validator, mock_system_validator):
        """Test successful validation returns specified and project directories."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_tracker = Mock()
        mock_feedback.return_value = mock_tracker
        
        args = Mock()
        args.directory = '/test/dir'
        
        mock_validator.validate_directory_only.return_value = Path('/test/dir')
        mock_validator.find_project_root.return_value = Path('/test')
        
        # Act
        with patch('smart_test_generator.cli.ProgressTracker', return_value=mock_tracker):
            specified_dir, project_dir = validate_system_and_project(args, mock_feedback)
        
        # Assert
        assert specified_dir == Path('/test/dir')
        assert project_dir == Path('/test')
        mock_system_validator.check_python_version.assert_called_once()
        mock_validator.check_dependencies.assert_called_once()
        mock_validator.validate_directory_only.assert_called_once_with('/test/dir', must_exist=True, must_be_writable=True)
        mock_validator.find_project_root.assert_called_once_with(Path('/test/dir'))
        mock_validator.validate_python_project.assert_called_once_with(Path('/test'))
    
    @patch('smart_test_generator.cli.SystemValidator')
    @patch('smart_test_generator.cli.Validator')
    @patch('sys.exit')
    def test_dependency_error_exits_with_error(self, mock_exit, mock_validator, mock_system_validator):
        """Test that DependencyError causes system exit."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_tracker = Mock()
        args = Mock()
        args.directory = '/test/dir'
        
        mock_system_validator.check_python_version.side_effect = DependencyError("Python version too old")
        
        # Act
        with patch('smart_test_generator.cli.ProgressTracker', return_value=mock_tracker):
            validate_system_and_project(args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()
    
    @patch('smart_test_generator.cli.SystemValidator')
    @patch('smart_test_generator.cli.Validator')
    @patch('sys.exit')
    def test_validation_error_exits_with_error(self, mock_exit, mock_validator, mock_system_validator):
        """Test that ValidationError causes system exit."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_tracker = Mock()
        args = Mock()
        args.directory = '/test/dir'
        
        mock_validator.validate_directory_only.side_effect = ValidationError("Directory not found")
        
        # Act
        with patch('smart_test_generator.cli.ProgressTracker', return_value=mock_tracker):
            validate_system_and_project(args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()
    
    @patch('smart_test_generator.cli.SystemValidator')
    @patch('smart_test_generator.cli.Validator')
    @patch('sys.exit')
    def test_project_structure_error_exits_with_error(self, mock_exit, mock_validator, mock_system_validator):
        """Test that ProjectStructureError causes system exit."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_tracker = Mock()
        args = Mock()
        args.directory = '/test/dir'
        
        mock_validator.validate_python_project.side_effect = ProjectStructureError("Not a Python project")
        
        # Act
        with patch('smart_test_generator.cli.ProgressTracker', return_value=mock_tracker):
            validate_system_and_project(args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()


class TestLoadAndValidateConfig:
    """Test load_and_validate_config function."""
    
    @patch('smart_test_generator.cli.Config')
    @patch('smart_test_generator.cli.Validator')
    def test_successful_config_loading(self, mock_validator, mock_config_class):
        """Test successful configuration loading."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            'test_generation.style.framework': 'pytest',
            'test_generation.coverage.minimum_line_coverage': 80,
            'test_generation.exclude_dirs': ['venv', '__pycache__']
        }.get(key, default)
        mock_config_class.return_value = mock_config
        
        args = Mock()
        args.config = '.testgen.yml'
        
        mock_validator.validate_config_file.return_value = Path('.testgen.yml')
        
        # Act
        result = load_and_validate_config(args, mock_feedback)
        
        # Assert
        assert result == mock_config
        mock_validator.validate_config_file.assert_called_once_with('.testgen.yml')
        mock_config_class.assert_called_once_with('.testgen.yml')
        mock_feedback.summary_panel.assert_called_once()
    
    @patch('smart_test_generator.cli.Config')
    @patch('sys.exit')
    def test_configuration_error_exits_with_error(self, mock_exit, mock_config_class):
        """Test that ConfigurationError causes system exit."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        args = Mock()
        args.config = '.testgen.yml'
        
        mock_config_class.side_effect = ConfigurationError("Invalid config")
        
        # Act
        load_and_validate_config(args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()
    
    @patch('smart_test_generator.cli.Config')
    def test_config_loading_without_config_attribute(self, mock_config_class):
        """Test config loading when args doesn't have config attribute."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_config = Mock()
        mock_config.get.return_value = 'pytest'
        mock_config_class.return_value = mock_config
        
        args = Mock(spec=[])
        
        # Act
        result = load_and_validate_config(args, mock_feedback)
        
        # Assert
        assert result == mock_config
        mock_config_class.assert_called_once_with(None)


class TestValidateArguments:
    """Test validate_arguments function."""
    
    @patch('smart_test_generator.cli.Validator')
    def test_successful_batch_size_validation(self, mock_validator):
        """Test successful batch size validation."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        args = Mock()
        args.batch_size = 5
        mock_validator.validate_batch_size.return_value = 5
        
        # Act
        validate_arguments(args, mock_feedback)
        
        # Assert
        mock_validator.validate_batch_size.assert_called_once_with(5)
        mock_feedback.debug.assert_called_once_with("Batch size validated: 5")
    
    @patch('smart_test_generator.cli.Validator')
    @patch('sys.exit')
    def test_validation_error_exits_with_error(self, mock_exit, mock_validator):
        """Test that ValidationError causes system exit."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        args = Mock()
        args.batch_size = -1
        mock_validator.validate_batch_size.side_effect = ValidationError("Invalid batch size")
        
        # Act
        validate_arguments(args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()
    
    def test_no_batch_size_attribute_does_nothing(self):
        """Test that missing batch_size attribute is handled gracefully."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        args = Mock(spec=[])
        
        # Act
        validate_arguments(args, mock_feedback)
        
        # Assert
        mock_feedback.debug.assert_not_called()
        mock_feedback.error.assert_not_called()


class TestExtractLlmCredentials:
    """Test extract_llm_credentials function."""
    
    def test_extracts_all_credentials_from_args(self):
        """Test that all LLM credentials are extracted from arguments."""
        # Arrange
        args = Mock()
        args.claude_api_key = 'claude-key'
        args.claude_model = 'claude-3-sonnet'
        args.endpoint = 'https://api.azure.com'
        args.api_key = 'azure-key'
        args.deployment = 'gpt-4'
        
        # Act
        result = extract_llm_credentials(args)
        
        # Assert
        expected = {
            'claude_api_key': 'claude-key',
            'claude_model': 'claude-3-sonnet',
            'azure_endpoint': 'https://api.azure.com',
            'azure_api_key': 'azure-key',
            'azure_deployment': 'gpt-4'
        }
        assert result == expected
    
    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'env-claude-key'})
    def test_uses_environment_variable_when_arg_is_none(self):
        """Test that environment variable is used when argument is None."""
        # Arrange
        args = Mock()
        args.claude_api_key = None
        args.claude_model = 'claude-3-sonnet'
        args.endpoint = None
        args.api_key = None
        args.deployment = None
        
        # Act
        result = extract_llm_credentials(args)
        
        # Assert
        assert result['claude_api_key'] == 'env-claude-key'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_no_credentials_available(self):
        """Test that None is returned when no credentials are available."""
        # Arrange
        args = Mock()
        args.claude_api_key = None
        args.claude_model = 'claude-3-sonnet'
        args.endpoint = None
        args.api_key = None
        args.deployment = None
        
        # Act
        result = extract_llm_credentials(args)
        
        # Assert
        assert result['claude_api_key'] is None
        assert result['azure_endpoint'] is None
        assert result['azure_api_key'] is None
        assert result['azure_deployment'] is None


class TestHandleInitConfigMode:
    """Test handle_init_config_mode function."""
    
    @patch('smart_test_generator.cli.Config')
    @patch('os.path.exists')
    def test_creates_new_config_file_successfully(self, mock_exists, mock_config_class):
        """Test successful creation of new configuration file."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_config = Mock()
        mock_config.create_sample_config.return_value = True
        mock_config_class.return_value = mock_config
        
        args = Mock()
        args.config = '.testgen.yml'
        
        mock_exists.return_value = False
        
        # Act
        handle_init_config_mode(args, mock_feedback)
        
        # Assert
        mock_config.create_sample_config.assert_called_once_with('.testgen.yml')
        mock_feedback.success.assert_called_once_with("Sample configuration created at .testgen.yml")
        mock_feedback.feature_showcase.assert_called_once()
    
    @patch('smart_test_generator.cli.Config')
    @patch('os.path.exists')
    def test_overwrites_existing_config_when_confirmed(self, mock_exists, mock_config_class):
        """Test overwriting existing config file when user confirms."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback.confirm.return_value = True
        mock_config = Mock()
        mock_config.create_sample_config.return_value = True
        mock_config_class.return_value = mock_config
        
        args = Mock()
        args.config = '.testgen.yml'
        
        mock_exists.return_value = True
        
        # Act
        handle_init_config_mode(args, mock_feedback)
        
        # Assert
        mock_feedback.warning.assert_called_once()
        mock_feedback.confirm.assert_called_once_with("Do you want to overwrite it?", default=False)
        mock_config.create_sample_config.assert_called_once_with('.testgen.yml')
    
    @patch('os.path.exists')
    def test_cancels_when_user_declines_overwrite(self, mock_exists):
        """Test cancellation when user declines to overwrite existing config."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback.confirm.return_value = False
        
        args = Mock()
        args.config = '.testgen.yml'
        
        mock_exists.return_value = True
        
        # Act
        handle_init_config_mode(args, mock_feedback)
        
        # Assert
        mock_feedback.confirm.assert_called_once_with("Do you want to overwrite it?", default=False)
        mock_feedback.info.assert_called_once_with("Configuration generation cancelled")
    
    @patch('smart_test_generator.cli.Config')
    @patch('os.path.exists')
    @patch('sys.exit')
    def test_exits_on_config_creation_error(self, mock_exit, mock_exists, mock_config_class):
        """Test system exit when config creation fails."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_config = Mock()
        mock_config.create_sample_config.side_effect = Exception("Permission denied")
        mock_config_class.return_value = mock_config
        
        args = Mock()
        args.config = '.testgen.yml'
        
        mock_exists.return_value = False
        
        # Act
        handle_init_config_mode(args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()


class TestExecuteModeWithStatus:
    """Test execute_mode_with_status function."""
    
    def test_status_mode_execution(self):
        """Test successful status mode execution."""
        # Arrange
        mock_app = Mock()
        mock_app.run_status_mode.return_value = "Status result"
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        
        args = Mock()
        args.mode = 'status'
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_app.run_status_mode.assert_called_once()
        mock_feedback.info.assert_called_with("Status result")
    
    def test_analyze_mode_execution(self):
        """Test successful analyze mode execution."""
        # Arrange
        mock_app = Mock()
        mock_app.run_analysis_mode.return_value = "Analysis result"
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        
        args = Mock()
        args.mode = 'analyze'
        args.force = True
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_app.run_analysis_mode.assert_called_once_with(force=True)
        mock_feedback.info.assert_called_with("Analysis result")
    
    def test_coverage_mode_execution(self):
        """Test successful coverage mode execution."""
        # Arrange
        mock_app = Mock()
        mock_app.run_coverage_mode.return_value = "Coverage result"
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        
        args = Mock()
        args.mode = 'coverage'
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_app.run_coverage_mode.assert_called_once()
        mock_feedback.info.assert_called_with("Coverage result")
    
    @patch('smart_test_generator.cli.extract_llm_credentials')
    @patch('sys.exit')
    def test_generate_mode_exits_without_claude_api_key(self, mock_exit, mock_extract):
        """Test that generate mode exits when Claude API key is missing."""
        # Arrange
        mock_app = Mock()
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_extract.return_value = {'claude_api_key': None}
        
        args = Mock()
        args.mode = 'generate'
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()
    
    @patch('smart_test_generator.cli.extract_llm_credentials')
    def test_generate_mode_execution_with_credentials(self, mock_extract):
        """Test successful generate mode execution with credentials."""
        # Arrange
        mock_app = Mock()
        mock_app.run_generate_mode.return_value = "Generation result"
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        
        credentials = {
            'claude_api_key': 'test-key',
            'claude_model': 'claude-3-sonnet'
        }
        mock_extract.return_value = credentials
        
        args = Mock()
        args.mode = 'generate'
        args.batch_size = 5
        args.force = False
        args.dry_run = True
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_app.run_generate_mode.assert_called_once_with(
            llm_credentials=credentials,
            batch_size=5,
            force=False,
            dry_run=True
        )
        mock_feedback.info.assert_called_with("Generation result")
    
    @patch('smart_test_generator.cli.CostManager')
    def test_cost_mode_execution(self, mock_cost_manager_class):
        """Test successful cost mode execution."""
        # Arrange
        mock_app = Mock()
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_cost_manager = Mock()
        mock_cost_manager.get_usage_summary.return_value = {
            'total_cost': 1.25,
            'requests': 10,
            'total_tokens': 5000,
            'average_cost_per_request': 0.125
        }
        mock_cost_manager_class.return_value = mock_cost_manager
        
        args = Mock()
        args.mode = 'cost'
        args.usage_days = 7
        
        # Act
        with patch('smart_test_generator.cli.config', Mock()):
            execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_cost_manager.get_usage_summary.assert_called_once_with(7)
        mock_feedback.summary_panel.assert_called_once()
    
    @patch('sys.exit')
    def test_smart_test_generator_error_exits_gracefully(self, mock_exit):
        """Test that SmartTestGeneratorError causes graceful exit."""
        # Arrange
        mock_app = Mock()
        mock_app.run_status_mode.side_effect = SmartTestGeneratorError("Test error")
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        
        args = Mock()
        args.mode = 'status'
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
    
    @patch('sys.exit')
    def test_unexpected_error_exits_with_traceback(self, mock_exit):
        """Test that unexpected errors cause exit with traceback."""
        # Arrange
        mock_app = Mock()
        mock_app.run_status_mode.side_effect = Exception("Unexpected error")
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback.verbose = True
        
        args = Mock()
        args.mode = 'status'
        
        # Act
        execute_mode_with_status(mock_app, args, mock_feedback)
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called()


class TestMain:
    """Test main function."""
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('smart_test_generator.cli.show_welcome_banner')
    @patch('smart_test_generator.cli.validate_system_and_project')
    @patch('smart_test_generator.cli.load_and_validate_config')
    @patch('smart_test_generator.cli.validate_arguments')
    @patch('smart_test_generator.cli.SmartTestGeneratorApp')
    @patch('smart_test_generator.cli.execute_mode_with_status')
    def test_successful_main_execution(
        self, mock_execute, mock_app_class, mock_validate_args,
        mock_load_config, mock_validate_system, mock_banner,
        mock_feedback_class, mock_setup_parser
    ):
        """Test successful main function execution."""
        # Arrange
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.mode = 'generate'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback_class.return_value = mock_feedback
        
        mock_validate_system.return_value = (Path('/test'), Path('/project'))
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        mock_app = Mock()
        mock_app_class.return_value = mock_app
        
        # Act
        main()
        
        # Assert
        mock_setup_parser.assert_called_once()
        mock_feedback_class.assert_called_once_with(verbose=False)
        mock_banner.assert_called_once_with(mock_feedback)
        mock_validate_system.assert_called_once_with(mock_args, mock_feedback)
        mock_load_config.assert_called_once_with(mock_args, mock_feedback)
        mock_validate_args.assert_called_once_with(mock_args, mock_feedback)
        mock_app_class.assert_called_once_with(Path('/test'), Path('/project'), mock_config, mock_feedback)
        mock_execute.assert_called_once_with(mock_app, mock_args, mock_feedback)
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('smart_test_generator.cli.handle_init_config_mode')
    def test_init_config_mode_exits_early(
        self, mock_handle_init, mock_feedback_class, mock_setup_parser
    ):
        """Test that init-config mode exits early without other setup."""
        # Arrange
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.mode = 'init-config'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback_class.return_value = mock_feedback
        
        # Act
        main()
        
        # Assert
        mock_handle_init.assert_called_once_with(mock_args, mock_feedback)
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('sys.exit')
    def test_keyboard_interrupt_exits_gracefully(self, mock_exit, mock_feedback_class, mock_setup_parser):
        """Test that KeyboardInterrupt exits gracefully."""
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_args.side_effect = KeyboardInterrupt()
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback_class.return_value = mock_feedback
        
        # Act
        main()
        
        # Assert
        mock_exit.assert_called_once_with(130)
        mock_feedback.warning.assert_called_once_with("Operation cancelled by user")
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('sys.exit')
    def test_smart_test_generator_error_exits_with_error(self, mock_exit, mock_feedback_class, mock_setup_parser):
        """Test that SmartTestGeneratorError exits with error code 1."""
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_args.side_effect = SmartTestGeneratorError("Test error")
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback_class.return_value = mock_feedback
        
        # Act
        main()
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called_once()
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('sys.exit')
    def test_unexpected_error_exits_with_error(self, mock_exit, mock_feedback_class, mock_setup_parser):
        """Test that unexpected errors exit with error code 1."""
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_args.side_effect = Exception("Unexpected error")
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback.verbose = True
        mock_feedback_class.return_value = mock_feedback
        
        # Act
        main()
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called()
    
    @patch('smart_test_generator.cli.setup_argparse')
    def test_keyboard_interrupt_without_feedback_prints_message(self, mock_setup_parser):
        """Test that KeyboardInterrupt without feedback prints message to stdout."""
        # Arrange
        mock_parser = Mock()
        mock_parser.parse_args.side_effect = KeyboardInterrupt()
        mock_setup_parser.return_value = mock_parser
        
        # Act
        with patch('sys.exit') as mock_exit, patch('builtins.print') as mock_print:
            main()
        
        # Assert
        mock_exit.assert_called_once_with(130)
        mock_print.assert_called_once_with("\nOperation cancelled by user")
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.SmartTestGeneratorApp')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('smart_test_generator.cli.validate_system_and_project')
    @patch('smart_test_generator.cli.load_and_validate_config')
    @patch('smart_test_generator.cli.validate_arguments')
    @patch('sys.exit')
    def test_app_initialization_error_exits_gracefully(
        self, mock_exit, mock_validate_args, mock_load_config,
        mock_validate_system, mock_feedback_class, mock_app_class, mock_setup_parser
    ):
        """Test that app initialization errors exit gracefully."""
        # Arrange
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.mode = 'generate'
        mock_args.verbose = False
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback_class.return_value = mock_feedback
        
        mock_validate_system.return_value = (Path('/test'), Path('/project'))
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        mock_app_class.side_effect = Exception("App init failed")
        
        # Act
        with patch('smart_test_generator.cli.show_welcome_banner'):
            main()
        
        # Assert
        mock_exit.assert_called_once_with(1)
        mock_feedback.error.assert_called()
    
    @patch('smart_test_generator.cli.setup_argparse')
    @patch('smart_test_generator.cli.UserFeedback')
    @patch('smart_test_generator.cli.show_welcome_banner')
    @patch('smart_test_generator.cli.validate_system_and_project')
    @patch('smart_test_generator.cli.load_and_validate_config')
    @patch('smart_test_generator.cli.validate_arguments')
    @patch('smart_test_generator.cli.SmartTestGeneratorApp')
    @patch('smart_test_generator.cli.execute_mode_with_status')
    def test_verbose_logging_is_enabled_when_verbose_flag_set(
        self, mock_execute, mock_app_class, mock_validate_args,
        mock_load_config, mock_validate_system, mock_banner,
        mock_feedback_class, mock_setup_parser
    ):
        """Test that verbose logging is enabled when verbose flag is set."""
        # Arrange
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.mode = 'generate'
        mock_args.verbose = True
        mock_parser.parse_args.return_value = mock_args
        mock_setup_parser.return_value = mock_parser
        
        mock_feedback = Mock()
        mock_feedback.status_spinner.return_value.__enter__ = Mock()
        mock_feedback.status_spinner.return_value.__exit__ = Mock(return_value=None)
        mock_feedback_class.return_value = mock_feedback
        
        mock_validate_system.return_value = (Path('/test'), Path('/project'))
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        mock_app = Mock()
        mock_app_class.return_value = mock_app
        
        # Act
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            main()
        
        # Assert
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
        mock_feedback.debug.assert_called_once_with("Verbose logging enabled")

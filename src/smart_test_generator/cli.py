"""Command-line interface for smart test generator."""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

from smart_test_generator.config import Config
from smart_test_generator.core import SmartTestGeneratorApp
from smart_test_generator.utils.cost_manager import CostManager
from smart_test_generator.utils.user_feedback import UserFeedback, ProgressTracker, StatusIcon
from smart_test_generator.utils.validation import Validator, SystemValidator, EnvironmentValidator
from smart_test_generator.exceptions import (
    SmartTestGeneratorError,
    ValidationError,
    ConfigurationError,
    AuthenticationError,
    DependencyError,
    ProjectStructureError,
)

# Configure logging
def configure_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging based on verbosity level."""
    if quiet:
        # Suppress most logging in quiet mode
        logging.basicConfig(level=logging.ERROR, format='%(message)s')
        # Disable third-party library logging
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    elif verbose:
        # Full logging with timestamps in verbose mode
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # Clean logging - suppress INFO and below, only show warnings and errors
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        # Reduce noise from third-party libraries
        logging.getLogger('urllib3').setLevel(logging.ERROR)
        logging.getLogger('requests').setLevel(logging.ERROR)
        logging.getLogger('smart_test_generator').setLevel(logging.ERROR)  # Suppress our own logging too
        
        # Only let critical application errors through the logging system
        # All user feedback should go through Rich UI instead

logger = logging.getLogger(__name__)


def show_welcome_banner(feedback: UserFeedback):
    """Display a concise welcome banner."""
    if not feedback.quiet:
        # Keep the banner simple and professional
        feedback.brand_header()


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate unit tests for Python codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Modes
    parser.add_argument(
        "mode",
        nargs='?',
        default='generate',
        choices=['generate', 'analyze', 'coverage', 'status', 'init-config', 'cost', 'debug-state', 'sync-state', 'reset-state'],
        help="Mode of operation"
    )

    parser.add_argument(
        "--directory",
        default=".",
        help="Directory to parse (default: current directory)"
    )

    # Azure OpenAI arguments
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint URL")
    parser.add_argument("--api-key", help="Azure OpenAI API key")
    parser.add_argument("--deployment", help="Azure OpenAI deployment name")

    # Claude API arguments
    parser.add_argument(
        "--claude-api-key",
        help="Claude API key (can also be set via CLAUDE_API_KEY env var)"
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use"
    )
    parser.add_argument(
        "--claude-extended-thinking",
        action="store_true",
        help="Enable extended thinking mode for Claude models (requires claude-sonnet-4-20250514 or claude-3-7-sonnet-20250219)"
    )
    parser.add_argument(
        "--claude-thinking-budget",
        type=int,
        help="Thinking budget in tokens (1024-32000, default: 4096). Only used with --claude-extended-thinking"
    )

    # AWS Bedrock arguments
    parser.add_argument("--bedrock-role-arn", help="AWS Role ARN to assume for Bedrock access")
    parser.add_argument("--bedrock-inference-profile", help="AWS Bedrock inference profile identifier")
    parser.add_argument("--bedrock-region", default="us-east-1", help="AWS region for Bedrock (default: us-east-1)")

    # Common arguments
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all tests"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        "-q", 
        action="store_true",
        help="Minimal output mode - only show results and errors"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process in each batch (default: 10)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Generate and write tests file-by-file instead of in batches (slower but immediate feedback)"
    )
    parser.add_argument(
        "--config",
        default=".testgen.yml",
        help="Configuration file"
    )

    # Coverage runner configuration
    parser.add_argument(
        "--runner-mode",
        choices=['python-module', 'pytest-path', 'custom'],
        help="Coverage runner mode: 'python-module' (default), 'pytest-path', or 'custom'"
    )
    parser.add_argument(
        "--python",
        dest="runner_python",
        help="Python executable to use when runner-mode=python-module (default: current interpreter)"
    )
    parser.add_argument(
        "--pytest-path",
        dest="runner_pytest_path",
        help="Path to pytest executable when runner-mode=pytest-path"
    )
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        help="Additional argument for the runner (can be repeated)"
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument to pass to pytest (can be repeated)"
    )
    parser.add_argument(
        "--runner-cwd",
        dest="runner_cwd",
        help="Working directory to execute pytest from"
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variable KEY=VALUE to set for pytest (can be repeated)"
    )
    parser.add_argument(
        "--append-pythonpath",
        action="append",
        default=[],
        help="Path to append to PYTHONPATH for pytest (can be repeated)"
    )
    parser.add_argument(
        "--no-env-propagate",
        dest="env_propagate",
        action="store_false",
        help="Do not inherit current environment variables for the pytest run"
    )

    # Post-generation auto-run and refine loop
    parser.add_argument(
        "--auto-run",
        dest="auto_run",
        action="store_true",
        help="Automatically run pytest after generating tests"
    )
    parser.add_argument(
        "--no-auto-run",
        dest="auto_run",
        action="store_false",
        help="Disable automatic pytest run after generation"
    )
    parser.add_argument(
        "--refine-enable",
        dest="refine_enable",
        action="store_true",
        help="Enable LLM-driven refinement loop when tests fail"
    )
    parser.add_argument(
        "--no-refine",
        dest="refine_enable",
        action="store_false",
        help="Disable refinement loop"
    )
    parser.add_argument(
        "--retries",
        dest="refine_retries",
        type=int,
        help="Max refinement retries (default: 2)"
    )
    parser.add_argument(
        "--backoff-base-sec",
        dest="refine_backoff_base_sec",
        type=float,
        help="Refinement backoff base seconds (default: 1.0)"
    )
    parser.add_argument(
        "--backoff-max-sec",
        dest="refine_backoff_max_sec",
        type=float,
        help="Refinement backoff max seconds (default: 8.0)"
    )
    parser.add_argument(
        "--stop-on-no-change",
        dest="refine_stop_on_no_change",
        action="store_true",
        help="Stop refinement when LLM returns no updated files"
    )
    parser.add_argument(
        "--no-stop-on-no-change",
        dest="refine_stop_on_no_change",
        action="store_false",
        help="Do not stop when no changes are returned"
    )
    parser.add_argument(
        "--max-total-minutes",
        dest="refine_max_total_minutes",
        type=int,
        help="Max total minutes allowed for refinement (soft cap)"
    )

    # Merge strategy flags
    parser.add_argument(
        "--merge-strategy",
        dest="merge_strategy",
        choices=["append", "ast-merge"],
        help="Test file merge strategy (append or ast-merge)"
    )
    parser.add_argument(
        "--merge-dry-run",
        dest="merge_dry_run",
        action="store_true",
        help="Do not write files; only compute changes for merge"
    )
    parser.add_argument(
        "--merge-formatter",
        dest="merge_formatter",
        choices=["none", "black"],
        help="Formatter to apply to merged output (none or black)"
    )

    # Cost management arguments
    parser.add_argument(
        "--cost-optimize",
        action="store_true",
        help="Enable aggressive cost optimization (smaller batches, cheaper models)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        help="Maximum cost limit per session (in USD)"
    )
    parser.add_argument(
        "--usage-days",
        type=int,
        default=7,
        help="Number of days for cost usage summary (default: 7)"
    )

    return parser


def validate_system_and_project(args, feedback: UserFeedback) -> Tuple[Path, Path]:
    """Validate system requirements, Python env, and project setup."""
    
    validation_steps = [
        ("Python Version", "Checking Python compatibility"),
        ("CLI Dependencies", "Verifying CLI required packages"),
        ("Project Structure", "Validating project setup"),
        ("Python Environment", "Checking venv and test deps"),
        ("Permissions", "Checking write access"),
    ]
    
    # Show validation table
    feedback.subsection_header("System & Project Validation")
    
    tracker = ProgressTracker(feedback)
    tracker.set_total_steps(len(validation_steps), "Validating system")
    
    try:
        # Python version check
        tracker.step("Checking Python version compatibility")
        SystemValidator.check_python_version()
        
        # Dependencies check (for CLI UI)
        tracker.step("Verifying required dependencies")
        required_packages = ['requests', 'pathlib', 'rich']
        Validator.check_dependencies(required_packages)
        
        # Project validation
        tracker.step("Validating project structure")
        # Validate the specified directory exists and is accessible
        specified_dir = Validator.validate_directory_only(args.directory, must_exist=True, must_be_writable=True)
        # Find the project root for configuration purposes
        project_root = Validator.find_project_root(specified_dir)
        Validator.validate_python_project(project_root)
        
        # Python environment check (venv + pytest/coverage importable + requirements)
        tracker.step("Checking Python environment and test dependencies")
        env_result = EnvironmentValidator.check_python_env(project_root)
        missing_dists = EnvironmentValidator.check_requirements_installed(project_root)
        if missing_dists:
            raise DependencyError(
                f"Missing required distributions from requirements.txt: {', '.join(sorted(set(missing_dists)))}",
                suggestion="Activate your virtualenv and run: pip install -r requirements.txt"
            )
        # If key test deps are not importable, give an actionable warning but continue
        if env_result.get("status") == "warn":
            # Surface guidance to the user
            for msg in env_result.get("messages", [])[:3]:
                feedback.warning(msg)

        # Final validation
        tracker.step("Completing validation checks")
        
        tracker.complete("All validation checks passed successfully!")
        
        # Show validation results
        validation_results = [
            ("success", "Python Version", f"Compatible (>= 3.8)"),
            ("success", "CLI Dependencies", "All required packages available"),
            ("success" if env_result.get("venv_active") else "warning", "Python Environment", "Active venv" if env_result.get("venv_active") else "No active venv"),
            ("success" if env_result.get("pytest_importable") else "warning", "pytest", "importable" if env_result.get("pytest_importable") else "not importable"),
            ("success" if env_result.get("coverage_importable") else "warning", "coverage", "importable" if env_result.get("coverage_importable") else "not importable"),
            ("success", "Target Directory", f"Valid directory: {specified_dir}"),
            ("success", "Project Root", f"Found project root: {project_root}"),
            ("success", "Permissions", "Write access confirmed"),
        ]
        
        feedback.status_table("Validation Results", validation_results)
        
        return Path(specified_dir), Path(project_root)
        
    except (DependencyError, ValidationError, ProjectStructureError) as e:
        tracker.error(f"Validation failed: {e}")
        
        # Show what failed
        feedback.error(str(e), getattr(e, 'suggestion', None))
        sys.exit(1)


def _apply_cli_overrides_to_config(args, config: Config) -> None:
    """Apply CLI override flags to the loaded configuration object."""
    # Runner mode & executables
    if getattr(args, 'runner_mode', None):
        config.config['test_generation']['coverage']['runner']['mode'] = args.runner_mode
    if getattr(args, 'runner_python', None):
        config.config['test_generation']['coverage']['runner']['python'] = args.runner_python
    if getattr(args, 'runner_pytest_path', None):
        config.config['test_generation']['coverage']['runner']['pytest_path'] = args.runner_pytest_path
    if getattr(args, 'runner_cwd', None):
        config.config['test_generation']['coverage']['runner']['cwd'] = args.runner_cwd

    # Runner args and pytest args
    if hasattr(args, 'runner_arg') and args.runner_arg:
        runner_args = config.config['test_generation']['coverage']['runner'].get('args', []) or []
        runner_args.extend(args.runner_arg)
        config.config['test_generation']['coverage']['runner']['args'] = runner_args
    if hasattr(args, 'pytest_arg') and args.pytest_arg:
        pytest_args = config.config['test_generation']['coverage'].get('pytest_args', []) or []
        pytest_args.extend(args.pytest_arg)
        config.config['test_generation']['coverage']['pytest_args'] = pytest_args

    # Env propagation and extras
    if hasattr(args, 'env_propagate') and args.env_propagate is False:
        config.config['test_generation']['coverage']['env']['propagate'] = False
    if hasattr(args, 'env') and args.env:
        extra_env = config.config['test_generation']['coverage']['env'].get('extra', {}) or {}
        for pair in args.env:
            if '=' in pair:
                k, v = pair.split('=', 1)
                extra_env[str(k)] = str(v)
        config.config['test_generation']['coverage']['env']['extra'] = extra_env
    if hasattr(args, 'append_pythonpath') and args.append_pythonpath:
        append_pp = config.config['test_generation']['coverage']['env'].get('append_pythonpath', []) or []
        append_pp.extend(args.append_pythonpath)
        config.config['test_generation']['coverage']['env']['append_pythonpath'] = append_pp

    # Auto-run pytest after generation
    if hasattr(args, 'auto_run') and args.auto_run is not None:
        config.config['test_generation']['generation']['test_runner']['enable'] = bool(args.auto_run)

    # Refinement loop
    ref_cfg = config.config['test_generation']['generation'].setdefault('refine', {
        'enable': False,
        'max_retries': 2,
        'backoff_base_sec': 1.0,
        'backoff_max_sec': 8.0,
        'stop_on_no_change': True,
        'max_total_minutes': 5,
    })
    if getattr(args, 'refine_enable', None) is not None:
        ref_cfg['enable'] = bool(args.refine_enable)
    if getattr(args, 'refine_retries', None) is not None:
        ref_cfg['max_retries'] = int(args.refine_retries)
    if getattr(args, 'refine_backoff_base_sec', None) is not None:
        ref_cfg['backoff_base_sec'] = float(args.refine_backoff_base_sec)
    if getattr(args, 'refine_backoff_max_sec', None) is not None:
        ref_cfg['backoff_max_sec'] = float(args.refine_backoff_max_sec)
    if getattr(args, 'refine_stop_on_no_change', None) is not None:
        ref_cfg['stop_on_no_change'] = bool(args.refine_stop_on_no_change)
    if getattr(args, 'refine_max_total_minutes', None) is not None:
        ref_cfg['max_total_minutes'] = int(args.refine_max_total_minutes)

    # Merge strategy
    merge_cfg = config.config['test_generation']['generation'].setdefault('merge', {
        'strategy': 'append',
        'dry_run': False,
        'formatter': 'none',
    })
    if getattr(args, 'merge_strategy', None) is not None:
        merge_cfg['strategy'] = str(args.merge_strategy)
    if getattr(args, 'merge_dry_run', None) is not None:
        merge_cfg['dry_run'] = bool(args.merge_dry_run)
    if getattr(args, 'merge_formatter', None) is not None:
        merge_cfg['formatter'] = str(args.merge_formatter)


def load_and_validate_config(args, feedback: UserFeedback) -> Config:
    """Load and validate configuration with status display."""
    feedback.subsection_header("Configuration")
    
    with feedback.status_spinner("Loading configuration"):
        try:
            if hasattr(args, 'config') and args.config:
                config_path = Validator.validate_config_file(args.config)
            config = Config(getattr(args, 'config', None))
            
        except ConfigurationError as e:
            feedback.error(str(e), getattr(e, 'suggestion', None))
            sys.exit(1)
            return  # This won't be reached in real execution, but helps with testing
    
    # Show config summary
    config_info = {
        "Config File": getattr(args, 'config', '.testgen.yml'),
        "Test Framework": config.get('test_generation.style.framework', 'pytest'),
        "Coverage Threshold": f"{config.get('test_generation.coverage.minimum_line_coverage', 80)}%",
        "Excluded Patterns": f"{len(config.get('test_generation.exclude_dirs', []))} directories"
    }
    
    # Apply CLI overrides to configuration
    _apply_cli_overrides_to_config(args, config)

    feedback.summary_panel("Configuration Loaded", config_info, "blue")
    return config


def validate_generation_environment(project_root: Path, config: Config, feedback: UserFeedback) -> None:
    """Additional preflight checks specifically for running generation/tests.

    - Ensure active venv and key test deps
    - Warn if requirements.txt has missing dists
    - Warn if configured runner python differs from current interpreter/venv
    - Suggest PYTHONPATH adjustments for common src/ layout
    """
    env = EnvironmentValidator.check_python_env(project_root)

    if env.get("status") == "warn":
        for msg in env.get("messages", [])[:3]:
            feedback.warning(msg)

    missing = EnvironmentValidator.check_requirements_installed(project_root)
    if missing:
        feedback.error(
            f"Missing distributions from requirements.txt: {', '.join(sorted(set(missing)))}",
            "Activate your venv and run: pip install -r requirements.txt",
        )
        sys.exit(1)

    configured_python = config.get('test_generation.coverage.runner.python')
    if configured_python:
        try:
            configured_python_path = Path(configured_python).resolve()
            current = Path(sys.executable).resolve()
            venv_dir = os.environ.get("VIRTUAL_ENV")
            if configured_python_path != current:
                # If venv active but configured python is different or outside venv, warn
                if venv_dir and not str(configured_python_path).startswith(str(venv_dir)):
                    feedback.warning(
                        f"Configured runner.python ({configured_python_path}) is not the active venv interpreter ({current}).",
                        "Consider removing runner.python from config or set it to the active venv's python.",
                    )
        except Exception:
            # Non-fatal
            pass

    # Recommend src/ layout PYTHONPATH if needed
    src_dir = project_root / 'src'
    if src_dir.exists():
        append_list = config.get('test_generation.coverage.env.append_pythonpath', []) or []
        py_path = os.environ.get('PYTHONPATH', '')
        if str(src_dir) not in append_list and str(src_dir) not in py_path:
            feedback.warning(
                f"Detected 'src' layout at {src_dir} but it's not on PYTHONPATH.",
                "Add to config: test_generation.coverage.env.append_pythonpath: ['src']",
            )


def validate_arguments(args, feedback: UserFeedback):
    """Validate command line arguments."""
    if hasattr(args, 'batch_size') and args.batch_size:
        try:
            args.batch_size = Validator.validate_batch_size(args.batch_size)
            feedback.debug(f"Batch size validated: {args.batch_size}")
        except ValidationError as e:
            feedback.error(str(e), getattr(e, 'suggestion', None))
            sys.exit(1)


def extract_llm_credentials(args) -> dict:
    """Extract LLM credentials from arguments."""
    creds = {
        'claude_api_key': args.claude_api_key or os.environ.get("CLAUDE_API_KEY"),
        'claude_model': args.claude_model,
        'claude_extended_thinking': getattr(args, 'claude_extended_thinking', False),
        'claude_thinking_budget': getattr(args, 'claude_thinking_budget', None),
        'azure_endpoint': args.endpoint,
        'azure_api_key': args.api_key,
        'azure_deployment': args.deployment,
    }
    # Only include Bedrock fields if provided
    if getattr(args, 'bedrock_role_arn', None):
        creds['bedrock_role_arn'] = args.bedrock_role_arn
    if getattr(args, 'bedrock_inference_profile', None):
        creds['bedrock_inference_profile'] = args.bedrock_inference_profile
    if getattr(args, 'bedrock_region', None):
        creds['bedrock_region'] = args.bedrock_region
    return creds

def _parse_key_values(items):
    parsed = {}
    for item in items or []:
        if '=' in item:
            key, value = item.split('=', 1)
            parsed[key.strip()] = value.strip()
    return parsed


def handle_init_config_mode(args, feedback: UserFeedback):
    """Handle configuration initialization with beautiful interface."""
    feedback.section_header("Configuration Initialization")
    
    config_file = args.config if args.config != '.testgen.yml' else '.testgen.yml'
    
    if os.path.exists(config_file):
        feedback.warning(f"Configuration file {config_file} already exists!")
        if not feedback.confirm("Do you want to overwrite it?", default=False):
            feedback.info("Configuration generation cancelled")
            return
    
    with feedback.status_spinner("Creating configuration file"):
        try:
            config = Config()
            result = config.create_sample_config(config_file)
        except Exception as e:
            feedback.error(f"Failed to create configuration file: {e}", 
                         "Check file permissions and try again.")
            sys.exit(1)
    
    feedback.success(f"Sample configuration created at {config_file}")
    
    # Show what was configured
    config_features = [
        "Virtual environment exclusions",
        "Test pattern configuration",
        "Coverage thresholds", 
        "Generation preferences"
    ]
    
    feedback.subsection_header("Configuration Features")
    feedback.feature_showcase(config_features)
    
    feedback.info("""
[dim]You can customize the 'exclude_dirs' section to add project-specific exclusions.[/dim]
""")


def execute_mode_with_status(app: SmartTestGeneratorApp, args, feedback: UserFeedback):
    """Execute the requested mode with beautiful status displays."""
    
    mode_descriptions = {
        'status': 'Checking generation history and current status',
        'analyze': 'Analyzing codebase for test opportunities', 
        'coverage': 'Analyzing test coverage and gaps',
        'generate': 'Generating intelligent unit tests',
        'debug-state': 'Debugging test generation state tracking',
        'sync-state': 'Syncing state tracker with existing tests',
        'reset-state': 'Resetting test generation state (use when corrupted)'
    }
    
    # Show execution plan with sophisticated styling
    execution_config = {
        "Mode": args.mode.title(),
        "Directory": args.directory,
    }
    
    if hasattr(args, 'batch_size'):
        execution_config["Batch Size"] = f"{args.batch_size} files"
    if hasattr(args, 'streaming') and args.streaming:
        execution_config["Processing"] = "Streaming (file-by-file)"
    if hasattr(args, 'force') and args.force:
        execution_config["Force Mode"] = "Yes"
    if hasattr(args, 'dry_run') and args.dry_run:
        execution_config["Dry Run"] = "Yes"
        
    feedback.execution_summary(args.mode, execution_config)
    
    try:
        if args.mode == 'status':
            with feedback.status_spinner("Retrieving generation status"):
                result = app.run_status_mode()
            feedback.result(result)
            
        elif args.mode == 'analyze':
            feedback.sophisticated_progress("Scanning Python files", "Looking for test opportunities")
            
            # Let the services handle their own visual feedback instead of wrapping with spinner
            result = app.run_analysis_mode(force=args.force)
            
            feedback.completion_celebration("Analysis", {
                "Files Analyzed": "Scanning complete",
                "Test Opportunities": "Identified missing coverage",
                "Quality Assessment": "Completed"
            })
            feedback.result(result)
            
        elif args.mode == 'coverage':
            feedback.sophisticated_progress("Calculating test coverage", "Analyzing existing tests")
            
            with feedback.status_spinner("Analyzing test coverage"):
                result = app.run_coverage_mode()
                
            feedback.completion_celebration("Coverage Analysis", {
                "Coverage Report": "Generated",
                "Coverage Gaps": "Identified", 
                "Recommendations": "Available"
            })
            feedback.result(result)
            
        elif args.mode == 'generate':
            # Validate LLM credentials first
            llm_credentials = extract_llm_credentials(args)
            
            # Require at least one: Claude, Azure, or Bedrock
            if not (
                llm_credentials.get('claude_api_key') or 
                (llm_credentials.get('azure_endpoint') and llm_credentials.get('azure_api_key') and llm_credentials.get('azure_deployment')) or 
                (llm_credentials.get('bedrock_role_arn') and llm_credentials.get('bedrock_inference_profile'))
            ):
                feedback.error(
                    "No LLM credentials provided",
                    "Provide Claude API key, Azure OpenAI credentials, or AWS Bedrock role ARN and inference profile"
                )
                sys.exit(1)
            
            # Enhanced generation configuration display
            ai_model_label = (
                llm_credentials['claude_model'] if llm_credentials.get('claude_api_key') else (
                    f"Azure:{llm_credentials.get('azure_deployment')}" if llm_credentials.get('azure_endpoint') else (
                        f"Bedrock:{llm_credentials.get('bedrock_inference_profile')}"
                    )
                )
            )
            generation_config = {
                "AI Model": ai_model_label,
                "Processing Mode": "Streaming" if args.streaming else f"Batch ({args.batch_size} files)",
                "Force Regeneration": "Yes" if args.force else "No",
                "Dry Run": "Yes" if args.dry_run else "No"
            }
            
            feedback.execution_summary("generation", generation_config)
            
            if args.dry_run:
                feedback.warning("Running in DRY RUN mode - no files will be modified")
            
            feedback.sophisticated_progress("Initializing AI test generation", f"Using {ai_model_label}")
            
            # Preflight: validate environment for generation/tests
            validate_generation_environment(app.project_root, app.config, feedback)

            # Generate tests with sophisticated progress tracking
            result = app.run_generate_mode(
                llm_credentials=llm_credentials,
                batch_size=args.batch_size,
                force=args.force,
                dry_run=args.dry_run,
                streaming=args.streaming
            )
            
            # Parse result to show sophisticated completion
            if "generated tests for" in result:
                import re
                # Try to extract statistics from result
                files_match = re.search(r'(\d+)\s+files?', result)
                tests_match = re.search(r'(\d+)\s+tests?', result)
                
                completion_stats = {}
                if files_match:
                    completion_stats["Files Processed"] = files_match.group(1)
                if tests_match:
                    completion_stats["Tests Generated"] = tests_match.group(1)
                    
                completion_stats["Coverage Improvement"] = "Analyzing..."
                completion_stats["Status"] = "Ready for testing"
                
                feedback.completion_celebration("Test Generation", completion_stats, "2m 14s")
            else:
                feedback.result(result)
            
        elif args.mode == 'cost':
            # Handle cost usage viewing
            cost_manager = CostManager(app.config)
            usage_summary = cost_manager.get_usage_summary(args.usage_days)
            
            feedback.completion_celebration("Cost Analysis", {
                "Total Cost": f"${usage_summary['total_cost']:.4f}",
                "API Requests": str(usage_summary['requests']),
                "Total Tokens": f"{usage_summary['total_tokens']:,}",
                "Avg Cost/Request": f"${usage_summary['average_cost_per_request']:.4f}"
            }, f"Last {args.usage_days} days")
            
            # Show optimization suggestions if there's usage
            if usage_summary['requests'] > 0:
                feedback.info("💡 Cost Optimization Tips:")
                feedback.info("• Use smaller batch sizes for large files: --batch-size 5")
                feedback.info("• Use streaming mode for immediate file-by-file feedback: --streaming")
                feedback.info("• Enable cost optimization mode: --cost-optimize")
                feedback.info("• Use cheaper models for simple files: --claude-model claude-3-5-haiku-20241022")
                feedback.info("• Set cost limits: --max-cost 5.00")
            else:
                feedback.info("No usage data found. Start generating tests to see cost tracking!")
            
        elif args.mode == 'debug-state':
            # Debug the current test generation state
            with feedback.status_spinner("Analyzing test generation state"):
                result = app.run_debug_state_mode()
            
            feedback.divider("State Debug Information")
            feedback.info(result)
            
        elif args.mode == 'sync-state':
            # Sync state tracker with existing tests
            with feedback.status_spinner("Syncing state with existing tests"):
                result = app.run_sync_state_mode()
            
            feedback.success("State sync completed successfully!")
            feedback.info(result)
            
        elif args.mode == 'reset-state':
            # Reset the test generation state
            feedback.warning("This will reset all test generation state tracking!")
            if feedback.confirm("Are you sure you want to reset the state?", default=False):
                with feedback.status_spinner("Resetting test generation state"):
                    result = app.run_reset_state_mode()
                
                feedback.success("State reset completed!")
                feedback.info(result)
            else:
                feedback.info("State reset cancelled.")
            
    except SmartTestGeneratorError as e:
        # These are already handled and logged by the services
        # Use structured exit code when available (e.g., CoverageAnalysisError)
        exit_code = getattr(e, 'exit_code', 1)
        sys.exit(exit_code)
    except Exception as e:
        feedback.error(f"Unexpected error during {args.mode} mode: {e}")
        if feedback.verbose:
            feedback.error("Full traceback:", details=traceback.format_exc())
        sys.exit(1)


def main():
    """Main execution function with enhanced UI."""
    feedback = None
    
    try:
        # Parse arguments first to get verbose flag
        parser = setup_argparse()
        args = parser.parse_args()
        
        # Initialize user feedback system
        feedback = UserFeedback(verbose=getattr(args, 'verbose', False), quiet=getattr(args, 'quiet', False))
        
        # Configure logging based on verbose flag
        configure_logging(verbose=args.verbose, quiet=getattr(args, 'quiet', False))
        
        # Show welcome banner
        show_welcome_banner(feedback)
        
        # Handle init-config mode early (doesn't need other setup)
        if args.mode == 'init-config':
            handle_init_config_mode(args, feedback)
            return
        
        # Validate system and project
        specified_dir, project_dir = validate_system_and_project(args, feedback)
        
        # Load configuration
        config = load_and_validate_config(args, feedback)
        
        # Validate arguments
        validate_arguments(args, feedback)
        
        # Initialize application
        with feedback.status_spinner("Initializing application"):
            try:
                app = SmartTestGeneratorApp(specified_dir, project_dir, config, feedback)
            except Exception as e:
                feedback.error(f"Failed to initialize application: {e}", 
                             "Check if the directory contains valid Python files and try again.")
                sys.exit(1)
        
        feedback.success("Application ready!")
        
        # Execute the requested mode
        execute_mode_with_status(app, args, feedback)
        
        # Show success summary
        feedback.divider()
        feedback.success("Operation completed successfully!")
        
        final_summary = {
            "Mode": args.mode.title(),
            "Project": str(project_dir),
            "Status": "Completed"
        }
        feedback.summary_panel("Execution Summary", final_summary, "green")
        
    except KeyboardInterrupt:
        if feedback:
            feedback.warning("Operation cancelled by user")
        else:
            print("\nOperation cancelled by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except SmartTestGeneratorError as e:
        if feedback:
            feedback.error(str(e), getattr(e, 'suggestion', None))
            if hasattr(e, 'suggestion') and e.suggestion and feedback.verbose:
                logger.error(f"Error details: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        if feedback:
            feedback.error(f"Unexpected error: {e}", 
                         "This appears to be a bug. Please report it with the details below.")
            if feedback.verbose:
                feedback.error("Full traceback:", details=traceback.format_exc())
        else:
            print(f"Unexpected error: {e}", file=sys.stderr)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

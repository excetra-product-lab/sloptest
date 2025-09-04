"""Configuration management for test generation."""

import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for test generation."""

    DEFAULT_CONFIG = {
        'test_generation': {
            'test_patterns': [
                'test_*.py',
                '*_test.py',
                'tests/**/test_*.py'
            ],
            'exclude': [
                'migrations/*',
                '*/deprecated/*',
                '__pycache__/*',
                '*.pyc'
            ],
            'exclude_dirs': [
                # Virtual environments
                'venv', 'env', '.env', '.venv', 'virtualenv',
                'ENV', 'env.bak', 'venv.bak',
                # Poetry virtual environments
                '.venv-*', 'poetry-*',
                # Conda environments
                'conda-meta', 'envs',
                # Pipenv environments
                '.venv-*',
                # Python build directories
                'build', 'dist', '*.egg-info', '*.dist-info',
                'pip-wheel-metadata', 'pip-build-env',
                # Cache directories
                '__pycache__', '.pytest_cache', '.coverage',
                '.cache', '.mypy_cache', '.ruff_cache',
                '.tox', '.nox', '.hypothesis',
                # IDE and editor directories
                '.vscode', '.idea', '.vs', '.atom',
                '.sublime-project', '.sublime-workspace',
                # Version control
                '.git', '.hg', '.svn', '.bzr',
                # Package managers
                'node_modules', 'bower_components',
                # Documentation build
                'docs/_build', '_build', 'site',
                # Testing and CI
                '.pytest_cache', '.coverage', 'htmlcov',
                '.stestr', '.testrepository',
                # Test generation artifacts - IMPORTANT: Exclude from LLM context
                '.artifacts',
                # Python-specific directories
                'site-packages', 'lib', 'lib64', 'include',
                'bin', 'Scripts', 'share', 'pyvenv.cfg',
                # OS-specific
                '.DS_Store', 'Thumbs.db',
                # Temporary directories
                'tmp', 'temp', '.tmp', '.temp',
                # Legacy Python
                'lib2to3', 'test', 'tests',
                # Jupyter
                '.ipynb_checkpoints',
                # Docker
                '.dockerignore', 'docker-compose.override.yml'
            ],
            'style': {
                'framework': 'pytest',
                'assertion_style': 'pytest',  # Options: 'pytest', 'unittest', 'auto'
                'mock_library': 'unittest.mock'  # Options: 'unittest.mock', 'pytest-mock', 'auto'
            },
            'coverage': {
                'minimum_line_coverage': 80,
                'minimum_branch_coverage': 70,  # Minimum branch coverage percentage
                'regenerate_if_below': 60,  # Regenerate tests if coverage drops below this
                # Additional runner configuration for coverage/pytest invocation
                'pytest_args': [],  # Extra arguments appended to pytest command
                'junit_xml': True,   # Always enable JUnit XML for all coverage runs (primary parsing method)
                'runner': {
                    'mode': 'python-module',          # 'python-module' | 'pytest-path' | 'custom'
                    'python': None,                   # If None, use current sys.executable
                    'pytest_path': 'pytest',          # Used when mode == 'pytest-path'
                    'custom_cmd': [],                 # Full custom argv when mode == 'custom'
                    'cwd': None,                      # Working directory; default project root
                    'args': []                        # Runner-provided args before pytest_args
                },
                'env': {
                    'propagate': True,                # Inherit current environment variables
                    'extra': {},                      # Extra env vars to set or override
                    'append_pythonpath': []           # Paths to append to PYTHONPATH
                }
            },
            'generation': {
                # Test content and structure options
                'include_docstrings': True,  # Include docstrings in test methods (true, false, "minimal")
                'generate_fixtures': True,  # Generate pytest fixtures for common setup
                'parametrize_similar_tests': True,  # Use @pytest.mark.parametrize for similar tests
                'max_test_methods_per_class': 20,  # Maximum test methods per class (0 for unlimited)
                'always_analyze_new_files': False,  # Always analyze new files even if they have tests
                
                # Post-generation pytest runner (Task 2.1)
                'test_runner': {
                    'enable': False,          # Default off to preserve current behavior
                    'args': [],               # Extra pytest args, e.g., ['-q']
                    'cwd': None,              # Working directory; default project root
                    'junit_xml': True         # Always enable JUnit XML for reliable failure parsing
                },
                'merge': {
                    'strategy': 'append',     # 'append' | 'ast-merge'
                    'dry_run': False,
                    'formatter': 'none'
                },
                'refine': {
                    'enable': False,
                    'max_retries': 2,
                    'backoff_base_sec': 1.0,
                    'backoff_max_sec': 8.0,
                    'stop_on_no_change': True,
                    'max_total_minutes': 5,
                    'strategy': 'auto'  # 'auto', 'comprehensive', 'balanced', 'dependency_focused', 'logic_focused', 'setup_focused'
                }
            }
        },
        'environment': {
            'auto_detect': True,                    # Auto-detect current environment manager
            'preferred_manager': 'auto',            # 'poetry' | 'pipenv' | 'conda' | 'uv' | 'venv' | 'auto'
            'respect_virtual_env': True,            # Always use current virtual env
            'dependency_validation': True,          # Validate deps before running tests
            
            # Environment-specific overrides
            'overrides': {
                'poetry': {
                    'use_poetry_run': True,         # Use `poetry run pytest` instead of direct python
                    'respect_poetry_venv': True,
                },
                'pipenv': {
                    'use_pipenv_run': True,         # Use `pipenv run pytest`
                },
                'conda': {
                    'activate_environment': True,   # Ensure conda environment is active
                },
                'uv': {
                    'use_uv_run': False,           # Use direct python instead of `uv run`
                }
            }
        },
        'cost_management': {
            'max_file_size_kb': 50,  # Skip files larger than 50KB for cost control
            'max_context_size_chars': 100000,  # Limit total context size
            'max_files_per_request': 15,  # Override batch size for large files
            'use_cheaper_model_threshold_kb': 10,  # Use cheaper model for files < 10KB
            'enable_content_compression': True,  # Remove comments and whitespace in prompts
            'cost_thresholds': {
                'daily_limit': 50.0,  # Maximum daily cost in USD
                'per_request_limit': 2.0,  # Maximum cost per request in USD
                'warning_threshold': 1.0   # Warn when request exceeds this cost
            },
            'skip_trivial_files': True,  # Skip files with < 5 functions/classes
            'token_usage_logging': True  # Log token usage for cost tracking
        },
        'security': {
            'enable_ast_validation': False,  # Use safer regex validation by default
            'max_generated_file_size': 50000,  # 50KB limit for generated test files
            'block_dangerous_patterns': True,  # Block potentially dangerous code patterns
        },
        
        # Quality Analysis Settings
        'quality': {
            'enable_quality_analysis': True,  # Enable quality analysis by default
            'enable_mutation_testing': True,  # Enable mutation testing by default
            'minimum_quality_score': 75.0,   # Minimum acceptable quality score
            'minimum_mutation_score': 80.0,  # Minimum acceptable mutation score
            'max_mutants_per_file': 50,      # Maximum mutants per file for performance
            'mutation_timeout': 30,          # Timeout in seconds for mutation testing
            'display_detailed_results': True,  # Show detailed quality analysis results
            
            # Modern Python Mutators (Task 6)
            'modern_mutators': {
                'enable_type_hints': True,      # Enable type hint mutations (Optional, Union, etc.)
                'enable_async_await': True,     # Enable async/await mutations
                'enable_dataclass': True,      # Enable dataclass mutations
                'type_hints_severity': 'medium',  # Severity for type hint mutations: low, medium, high
                'async_severity': 'high',         # Async mutations often critical for correctness
                'dataclass_severity': 'medium',   # Dataclass mutations typically medium severity
            }
        },
        
        # Prompt Engineering Settings (Based on latest 2024-2025 Guidelines)
        'prompt_engineering': {
            'use_2025_guidelines': True,  # Use improved prompts following latest best practices
            'encourage_step_by_step': True,  # Include step-by-step reasoning prompts (legacy)
            'use_positive_negative_examples': True,  # Include positive/negative examples in prompts
            'minimize_xml_structure': True,  # Reduce excessive XML tags in prompts
            'decisive_recommendations': True,  # Encourage single, strong recommendations
            'preserve_uncertainty': False,  # Whether to include hedging language (usually False for technical tasks)
            
            # Enhanced 2024-2025 Features
            'use_enhanced_reasoning': True,  # Use advanced Chain-of-Thought reasoning
            'enable_self_debugging': True,  # Enable self-debugging and review checkpoints
            'use_enhanced_examples': True,  # Use detailed examples with reasoning
            'enable_failure_strategies': True,  # Use failure-specific debugging strategies
            'confidence_based_adaptation': True,  # Adapt prompts based on confidence levels
            'track_reasoning_quality': True,  # Monitor and track reasoning quality
        }
    }

    def __init__(self, config_file: str = ".testgen.yml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        # When config_file is None or falsy, skip file I/O and return defaults
        if not self.config_file:
            return self.DEFAULT_CONFIG.copy()
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults; yaml.safe_load can return None
                    if not user_config:
                        return self.DEFAULT_CONFIG.copy()
                    return self._deep_merge(self.DEFAULT_CONFIG, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
        return self.DEFAULT_CONFIG.copy()

    def create_sample_config(self, filepath: str = ".testgen.yml") -> None:
        """Create a comprehensive configuration file with all options and explanations."""
        
        # Generate comprehensive config with ALL options and explanations
        config_content = """# SlopTest Configuration
# This file contains ALL available configuration options with detailed explanations.
# Uncomment and modify the sections you want to customize.

test_generation:
  # =============================================================================
  # TEST DISCOVERY PATTERNS
  # =============================================================================
  
  # Patterns for finding test files (supports glob patterns)
  test_patterns:
    - 'test_*.py'
    - '*_test.py'
    - 'tests/**/test_*.py'
  
  # Files and patterns to exclude from test generation
  exclude:
    - 'migrations/*'
    - '*/deprecated/*'
    - '__pycache__/*'
    - '*.pyc'
  
  # Directories to exclude from scanning (add your custom exclusions here)
  # Note: Virtual environments, build dirs, etc. are excluded by default
  exclude_dirs:
    # Add custom directories here:
    # - 'my_custom_dir'
    # - 'vendor_code'
    
    # The following are already excluded by default:
    # Virtual environments: venv, env, .venv, .env, virtualenv, ENV
    # Build directories: build, dist, __pycache__, *.egg-info
    # Cache directories: .pytest_cache, .coverage, .cache, .mypy_cache
    # IDE directories: .vscode, .idea, .vs, .atom
    # Version control: .git, .hg, .svn, .bzr
    # Artifacts: .artifacts (test run outputs, should not be in LLM context)

  # =============================================================================
  # TEST GENERATION STYLE
  # =============================================================================
  
  style:
    framework: 'pytest'              # Options: 'pytest', 'unittest'
    assertion_style: 'pytest'       # Options: 'pytest', 'unittest', 'auto'
    mock_library: 'unittest.mock'   # Options: 'unittest.mock', 'pytest-mock', 'auto'

  # =============================================================================
  # COVERAGE ANALYSIS & THRESHOLDS
  # =============================================================================
  
  coverage:
    # Coverage thresholds
    minimum_line_coverage: 80        # Minimum line coverage percentage
    minimum_branch_coverage: 70      # Minimum branch coverage percentage
    regenerate_if_below: 60         # Regenerate tests if coverage drops below this
    
    # Additional pytest arguments for coverage runs
    pytest_args: []                 # e.g., ['-v', '--tb=short']
    
    # Test runner configuration
    runner:
      mode: 'python-module'         # Options: 'python-module', 'pytest-path', 'custom'
      python: null                  # Python executable (null = current sys.executable)
      pytest_path: 'pytest'        # Path to pytest when mode is 'pytest-path'
      custom_cmd: []                # Custom command when mode is 'custom'
      cwd: null                     # Working directory (null = project root)
      args: []                      # Runner-specific args before pytest_args
    
    # Environment configuration for test runs
    env:
      propagate: true               # Inherit current environment variables
      extra: {}                     # Additional environment variables
      append_pythonpath: []         # Paths to append to PYTHONPATH

  # =============================================================================
  # TEST GENERATION BEHAVIOR
  # =============================================================================
  
  generation:
    # Test content and structure options
    include_docstrings: true         # Include docstrings in test methods (true, false, "minimal")
    generate_fixtures: true          # Generate pytest fixtures for common setup
    parametrize_similar_tests: true  # Use @pytest.mark.parametrize for similar tests
    max_test_methods_per_class: 20   # Maximum test methods per class (0 for unlimited)
    always_analyze_new_files: false  # Always analyze new files even if they have tests
    
    # Post-generation test runner (runs pytest after generating tests)
    test_runner:
      enable: false                 # Enable post-generation test execution
      args: []                      # Extra pytest args, e.g., ['-q', '-x']
      cwd: null                     # Working directory (null = project root)
      junit_xml: false              # Generate JUnit XML for failure parsing
    
    # Test merging strategy
    merge:
      strategy: 'append'            # Options: 'append', 'ast-merge'
      dry_run: false                # Preview changes without applying
      formatter: 'none'             # Code formatter to apply after merge
    
    # Test refinement loop (AI-powered test fixing)
    refine:
      enable: false                 # Enable AI-powered test refinement
      max_retries: 2                # Maximum refinement attempts
      backoff_base_sec: 1.0         # Base delay between refinement attempts
      backoff_max_sec: 8.0          # Maximum delay between attempts
      stop_on_no_change: true       # Stop if LLM returns no changes
      max_total_minutes: 5          # Maximum total time for refinement
      strategy: 'auto'              # Refinement strategy: 'auto' (detect based on failures),
                                    # 'comprehensive' (thorough approach for complex issues),
                                    # 'balanced' (general-purpose approach),
                                    # 'dependency_focused' (for import/dependency errors),
                                    # 'logic_focused' (for assertion/logic errors),
                                    # 'setup_focused' (for fixture/mock errors)
      
      # Git diff integration settings
      git_context:
        enable: true                # Include git context in refinement
        days_back: 7                # How many days back to look for changes
        max_commits: 5              # Maximum number of recent commits to analyze
        include_diff_content: false # Include actual diff content (uses more tokens)
        analyze_changed_sources: true  # Analyze source files that might affect tests

# =============================================================================
# ENVIRONMENT DETECTION & MANAGEMENT
# =============================================================================

environment:
  # Environment detection settings
  auto_detect: true                  # Auto-detect current environment manager
  preferred_manager: 'auto'          # 'poetry' | 'pipenv' | 'conda' | 'uv' | 'venv' | 'auto'
  respect_virtual_env: true          # Always use current virtual env
  dependency_validation: true        # Validate deps before running tests
  
  # Environment-specific overrides
  overrides:
    poetry:
      use_poetry_run: true           # Use `poetry run pytest` instead of direct python
      respect_poetry_venv: true
    pipenv:
      use_pipenv_run: true           # Use `pipenv run pytest`
    conda:
      activate_environment: true     # Ensure conda environment is active
    uv:
      use_uv_run: false             # Use direct python instead of `uv run`

# =============================================================================
# COST MANAGEMENT & OPTIMIZATION
# =============================================================================

cost_management:
  # File size limits for cost control
  max_file_size_kb: 50              # Skip files larger than this (KB)
  max_context_size_chars: 100000    # Limit total context size
  max_files_per_request: 15         # Override batch size for large files
  use_cheaper_model_threshold_kb: 10 # Use cheaper model for files < this size
  enable_content_compression: true   # Remove comments/whitespace in prompts
  
  # Cost thresholds and limits
  cost_thresholds:
    daily_limit: 50.0               # Maximum daily cost in USD
    per_request_limit: 2.0          # Maximum cost per request in USD
    warning_threshold: 1.0          # Warn when request exceeds this cost
  
  # Additional optimizations
  skip_trivial_files: true          # Skip files with < 5 functions/classes
  token_usage_logging: true         # Log token usage for cost tracking

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

security:
  enable_ast_validation: false      # Use AST validation (slower but more secure)
  max_generated_file_size: 50000    # Maximum size for generated test files (bytes)
  block_dangerous_patterns: true    # Block potentially dangerous code patterns

# =============================================================================
# PROMPT ENGINEERING (Advanced AI Settings)
# =============================================================================

prompt_engineering:
  use_2025_guidelines: true         # Use latest prompt best practices
  encourage_step_by_step: true      # Include step-by-step reasoning prompts (legacy)
  use_positive_negative_examples: true # Include positive/negative examples in prompts
  minimize_xml_structure: true      # Reduce excessive XML tags in prompts
  decisive_recommendations: true    # Encourage single, strong recommendations
  preserve_uncertainty: false       # Include hedging language (usually false)
  
  # Enhanced 2024-2025 Features
  use_enhanced_reasoning: true      # Use advanced Chain-of-Thought reasoning
  enable_self_debugging: true       # Enable self-debugging and review checkpoints
  use_enhanced_examples: true       # Use detailed examples with reasoning
  enable_failure_strategies: true   # Use failure-specific debugging strategies
  confidence_based_adaptation: true # Adapt prompts based on confidence levels
  track_reasoning_quality: true     # Monitor and track reasoning quality

# =============================================================================
# TEST QUALITY ANALYSIS
# =============================================================================

quality:
  # Quality analysis settings
  enable_quality_analysis: true      # Enable quality analysis by default
  enable_mutation_testing: true      # Enable mutation testing by default
  minimum_quality_score: 75.0        # Minimum acceptable quality score (%)
  minimum_mutation_score: 80.0       # Minimum acceptable mutation score (%)
  max_mutants_per_file: 50           # Maximum mutants per file for performance
  mutation_timeout: 30               # Timeout in seconds for mutation testing
  display_detailed_results: true     # Show detailed quality analysis results
  enable_pattern_analysis: true      # Enable failure pattern analysis for smart refinement

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Enable post-generation test running:
#   test_generation.generation.test_runner.enable: true
#   test_generation.generation.test_runner.args: ['-v']
#
# Enable AI-powered test refinement:
#   test_generation.generation.refine.enable: true
#   test_generation.generation.refine.max_retries: 3
#
# Add custom exclusions:
#   test_generation.exclude_dirs:
#     - 'vendor'
#     - 'third_party'
#
# Adjust cost limits:
#   cost_management.daily_limit: 25.0
#   cost_management.max_file_size_kb: 30
#
# =============================================================================
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"Comprehensive configuration created at {filepath}")
        logger.info("The config file includes ALL available options with detailed explanations")
        logger.info("Uncomment and modify the sections you want to customize")

    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value

    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Deeply merge user config with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

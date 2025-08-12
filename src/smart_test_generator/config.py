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
                'assertion_style': 'assert',
                'mock_library': 'unittest.mock'
            },
            'coverage': {
                'minimum_line_coverage': 80,
                'minimum_branch_coverage': 70,
                'regenerate_if_below': 60,  # Regenerate tests if coverage drops below this
                # Additional runner configuration for coverage/pytest invocation
                'pytest_args': [],  # Extra arguments appended to pytest command
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
                'include_docstrings': True,
                'generate_fixtures': True,
                'parametrize_similar_tests': True,
                'max_test_methods_per_class': 20,
                'always_analyze_new_files': True,
                # Post-generation pytest runner (Task 2.1)
                'test_runner': {
                    'enable': False,          # Default off to preserve current behavior
                    'args': [],               # Extra pytest args, e.g., ['-q']
                    'cwd': None,              # Working directory; default project root
                    'junit_xml': False        # When true, write JUnit XML to artifacts for failure parsing
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
                    'max_total_minutes': 5
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
        
        # Prompt Engineering Settings (Based on Anthropic 2025 Guidelines)
        'prompt_engineering': {
            'use_2025_guidelines': True,  # Use improved prompts following Anthropic's 2025 best practices
            'encourage_step_by_step': True,  # Include step-by-step reasoning prompts
            'use_positive_negative_examples': True,  # Include ✓/✗ examples in prompts
            'minimize_xml_structure': True,  # Reduce excessive XML tags in prompts
            'decisive_recommendations': True,  # Encourage single, strong recommendations
            'preserve_uncertainty': False,  # Whether to include hedging language (usually False for technical tasks)
        }
    }

    def __init__(self, config_file: str = ".testgen.yml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    return self._deep_merge(self.DEFAULT_CONFIG, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
        return self.DEFAULT_CONFIG.copy()

    def create_sample_config(self, filepath: str = ".testgen.yml") -> None:
        """Create a sample configuration file with all options."""
        sample_config = {
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
                    # Add custom directories to exclude here
                    'my_custom_venv',
                    'local_build',
                    # Virtual environments (already covered by defaults)
                    # 'venv', 'env', '.venv', '.env', 'virtualenv',
                    # Build directories (already covered by defaults) 
                    # 'build', 'dist', '__pycache__'
                ],
                'style': {
                    'framework': 'pytest',  # or 'unittest'
                    'assertion_style': 'assert',  # or 'self.assert'
                    'mock_library': 'unittest.mock'  # or 'pytest-mock'
                },
                'coverage': {
                    'minimum_line_coverage': 80,
                    'minimum_branch_coverage': 70,
                    'regenerate_if_below': 60
                },
                'generation': {
                    'include_docstrings': True,
                    'generate_fixtures': True,
                    'parametrize_similar_tests': True,
                    'max_test_methods_per_class': 20,
                    'always_analyze_new_files': True
                }
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Sample configuration created at {filepath}")
        logger.info("You can customize the exclude_dirs section to add project-specific exclusions")

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

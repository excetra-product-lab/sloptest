# SlopTest

<div align="center">

🤖 **AI-Powered Python Test Generation**

*Intelligently generates missing tests by analyzing existing coverage and test patterns*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## ✨ Features

- 🎯 **Intelligent Test Gap Detection** - Analyzes your codebase to identify exactly what needs testing
- 🤖 **Multi-Provider AI Integration** - Supports Claude API, Azure OpenAI, and AWS Bedrock
- 📊 **Coverage-Driven Generation** - Uses existing test coverage to generate only missing tests
- 🔄 **Automatic Refinement** - Self-healing tests that fix themselves when they fail
- 💰 **Cost Optimization** - Built-in cost tracking and optimization strategies
- 🎨 **Rich Terminal UI** - Beautiful, clean interface with progress tracking
- ⚡ **Batch Processing** - Efficient processing with configurable batch sizes
- 🔧 **Flexible Configuration** - YAML-based configuration with CLI overrides

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/excetra-product-lab/sloptest.git
cd sloptest

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Generate tests for your project
sloptest generate

# Use with Claude API
sloptest generate --claude-api-key="your-key-here"

# Use with Azure OpenAI
sloptest generate --endpoint="https://your-endpoint.openai.azure.com/" \
                       --api-key="your-key" \
                       --deployment="your-deployment"

# Use with AWS Bedrock
sloptest generate --bedrock-role-arn="arn:aws:iam::account:role/role-name" \
                       --bedrock-inference-profile="your-profile-id"
```

## 📖 Usage

### Available Modes

| Mode | Description |
|------|-------------|
| `generate` | Generate missing unit tests (default) |
| `analyze` | Analyze codebase and show test coverage gaps |
| `coverage` | Run coverage analysis and show detailed report |
| `status` | Show current project status and configuration |
| `init-config` | Initialize configuration file |
| `cost` | Display cost usage summary |

### Generation Options

```bash
# Basic generation
sloptest generate

# Quiet mode (minimal output)
sloptest generate -q

# Verbose mode (detailed logging)
sloptest generate -v

# Custom batch size
sloptest generate --batch-size=5

# Streaming mode (file-by-file processing)
sloptest generate --streaming

# Force regeneration of all tests
sloptest generate --force

# Dry run (show what would be done)
sloptest generate --dry-run
```

### Coverage Analysis

```bash
# Analyze test coverage gaps
sloptest analyze

# Run coverage analysis
sloptest coverage

# Custom coverage configuration
sloptest generate --runner-mode=pytest-path \
                       --pytest-path="/path/to/pytest" \
                       --pytest-arg="--tb=short"
```

### Auto-Run and Refinement

```bash
# Automatically run tests after generation
sloptest generate --auto-run

# Enable automatic test refinement when tests fail
sloptest generate --auto-run --refine-enable

# Configure refinement parameters
sloptest generate --auto-run --refine-enable \
                       --retries=3 \
                       --max-total-minutes=10
```

### Cost Management

```bash
# Enable cost optimization
sloptest generate --cost-optimize

# Set maximum cost limit
sloptest generate --max-cost=10.00

# View cost usage summary
sloptest cost --usage-days=30
```

## ⚙️ Configuration

### Configuration File

Create a `.testgen.yml` file in your project root:

```yaml
# AI Provider Configuration
ai:
  provider: "claude"  # claude, azure, bedrock
  claude:
    api_key: "${CLAUDE_API_KEY}"
    model: "claude-sonnet-4-20250514"
  azure:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    deployment: "gpt-4"
  bedrock:
    role_arn: "${AWS_BEDROCK_ROLE_ARN}"
    inference_profile: "${AWS_BEDROCK_INFERENCE_PROFILE}"
    region: "us-east-1"

# Test Generation Settings
test_generation:
  batch_size: 10
  streaming: false
  auto_run: true
  
  # Coverage Configuration
  coverage:
    runner:
      mode: "python-module"  # python-module, pytest-path, custom
      python: "python"
      cwd: "."
    pytest_args: ["--tb=short"]
    
  # Refinement Settings
  refine:
    enable: true
    retries: 2
    max_total_minutes: 15
    stop_on_no_change: true

# Cost Management
cost:
  optimize: false
  max_per_session: 50.00
  
# Output Settings
output:
  quiet: false
  verbose: false
  
# Security Settings
security:
  block_dangerous_patterns: true
  max_generated_file_size: 50000
```

### Environment Variables

```bash
# Claude API
export CLAUDE_API_KEY="your-claude-api-key"

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-azure-api-key"

# AWS Bedrock (requires AWS CLI configuration)
export AWS_BEDROCK_ROLE_ARN="arn:aws:iam::account:role/role-name"
export AWS_BEDROCK_INFERENCE_PROFILE="your-profile-id"
```

## 🎯 Advanced Features

### Test Merge Strategies

```bash
# Append new tests to existing files
sloptest generate --merge-strategy=append

# Use AST-based intelligent merging
sloptest generate --merge-strategy=ast-merge

# Format merged output with Black
sloptest generate --merge-strategy=ast-merge --merge-formatter=black

# Dry run merge (see changes without writing)
sloptest generate --merge-dry-run
```

### Custom Test Patterns

The tool automatically detects:
- 🔍 Functions and methods that need testing
- 📦 Classes requiring comprehensive test coverage
- 🚨 Error conditions and edge cases
- 🔄 Async functions and decorators
- 📊 Data models and dataclasses

### Coverage Integration

SlopTest integrates seamlessly with:
- ✅ **pytest** with pytest-cov
- ✅ **coverage.py**
- ✅ **Custom test runners**

## 📊 Output Modes

### Normal Mode (Default)
- Clean, professional output with Rich UI
- Progress indicators and status tables
- Beautiful panels for configuration and results

### Quiet Mode (`-q`)
- Minimal output - only essential progress and results
- Perfect for scripts or CI/CD pipelines
- No decorative elements or banners

### Verbose Mode (`-v`)
- Full diagnostic information
- Debug messages and detailed logging
- Complete transparency for troubleshooting

## 🔧 Troubleshooting

### Common Issues

**API Authentication Errors**
```bash
# Verify your API credentials
sloptest status

# Test with verbose logging
sloptest generate -v
```

**Coverage Issues**
```bash
# Check pytest configuration
sloptest coverage -v

# Test custom runner setup
sloptest generate --runner-mode=pytest-path --pytest-path="$(which pytest)"
```

**Generation Quality Issues**
```bash
# Enable refinement loop
sloptest generate --auto-run --refine-enable

# Use smaller batch sizes
sloptest generate --batch-size=3

# Try streaming mode
sloptest generate --streaming
```

### Debug Commands

```bash
# Show current configuration
sloptest status

# Debug state management
sloptest debug-state

# Reset state if needed
sloptest reset-state

# Sync state
sloptest sync-state
```

## 🏗️ Architecture

```
sloptest/
├── src/smart_test_generator/
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration management
│   ├── core/
│   │   ├── application.py        # Main application orchestrator
│   │   └── llm_factory.py        # LLM client factory
│   ├── analysis/
│   │   ├── code_analyzer.py      # Code analysis and parsing
│   │   ├── coverage_analyzer.py  # Coverage analysis
│   │   └── test_mapper.py        # Test-to-source mapping
│   ├── generation/
│   │   ├── llm_clients.py        # AI provider integrations
│   │   └── test_generator.py     # Test generation logic
│   ├── services/
│   │   ├── analysis_service.py   # Analysis orchestration
│   │   ├── coverage_service.py   # Coverage operations
│   │   └── test_generation_service.py  # Generation orchestration
│   └── utils/
│       ├── cost_manager.py       # Cost tracking
│       ├── user_feedback.py      # Rich UI components
│       └── validation.py         # Input validation
└── tests/                        # Comprehensive test suite
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for Python developers who want better test coverage
- Powered by state-of-the-art AI models from Anthropic, OpenAI, and AWS
- Inspired by the need for intelligent, automated testing solutions

---

<div align="center">

**SlopTest** - Making comprehensive test coverage effortless

[Report Bug](https://github.com/excetra-product-lab/sloptest/issues) · [Request Feature](https://github.com/excetra-product-lab/sloptest/issues) · [Documentation](https://github.com/excetra-product-lab/sloptest/wiki)

</div>

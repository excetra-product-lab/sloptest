# SlopTest

AI-assisted pytest generation for Python. It analyzes your code and current coverage, then writes only the missing tests. Supports Claude API, Azure OpenAI, and AWS Bedrock.

Key properties
- Coverage-driven: decides what to generate from real or AST-estimated coverage
- Incremental: generates only untested functions/methods
- Safe writers: append or AST-merge into tests/ with optional Black formatting
- Optional post-run + LLM refinement loop to fix failing tests
- Modern Python mutation testing: type hints, async/await, and dataclass patterns
- Quality analysis with comprehensive test strength evaluation

Install (with uv)
```bash
# create and activate a virtualenv (optional but recommended)
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install
uv pip install -e .

# development extras
uv pip install -e .[dev]
```

Quick start
```bash
# Claude API
export CLAUDE_API_KEY=...
sloptest generate

# Azure OpenAI
sloptest generate \
  --endpoint https://<resource>.openai.azure.com \
  --api-key <key> \
  --deployment <deployment>

# AWS Bedrock (assume-role + inference profile)
sloptest generate \
  --bedrock-role-arn arn:aws:iam::<account>:role/<role> \
  --bedrock-inference-profile <profile> \
  --bedrock-region us-east-1
```

How it works

Overview
```mermaid
flowchart TD
  A[sloptest CLI] --> B[Validate system & project]
  B --> C[Load config + apply CLI overrides]
  C --> D[Initialize App]
  D --> E[Sync state with existing tests]
  E --> F[Find Python files]
  F --> G{Mode}
  G -->|analyze| H[Analyze plans + show gaps]
  G -->|coverage| I[Run coverage + report]
  G -->|status| J[Show generation history]
  G -->|generate| K[Generate tests]

  subgraph Generation
    K --> L[Coverage analysis]
    L --> M{coverage path}
    M -->|pytest| N[Parse .coverage -> per-file metrics]
    M -->|AST fallback| O[Estimate coverage from AST + tests]
    N --> P[Create plans]
    O --> P
    P --> Q{streaming?}
    Q -->|yes| R[Per file: LLM -> write]
    Q -->|no| S[Batch N: LLM -> write]
    R --> T[Optional pytest]
    S --> T
    T --> U{failures?}
    U -->|no| V[Report + update state]
    U -->|yes| W[Refinement loop]
    W --> T
    V --> X[Done]
  end
```

Coverage path
```mermaid
flowchart LR
  A[CoverageService] --> B[CoverageAnalyzer]
  B --> C{pytest available}
  C -->|yes| D[build_pytest_command]
  D --> E[run_pytest]
  E --> F{.coverage exists}
  F -->|yes| G[Parse with coverage.py]
  F -->|no| H[Parse errors + guidance]
  H --> I[Fallback to AST]
  C -->|no| I[ASTCoverageAnalyzer]
  I --> J[Executable lines + functions]
  J --> K[Estimate covered lines/functions]
```

Planning and generation (per file)
```mermaid
flowchart LR
  A[PythonCodebaseParser] --> B[CodeAnalyzer - testable elements]
  A --> C[TestMapper - existing tests]
  B --> D[TestGenerationPlan]
  C --> D
  D --> E[IncrementalLLMClient]
  E --> F[Prompt - dir tree, TODO XML, imports and signatures]
  F --> G[LLMClient - providers Claude Azure Bedrock]
  G --> H[JSON mapping source_py to tests]
  H --> I[TestFileWriter]
  I --> J[Test files under tests folder]
  J --> K[TestGenerationTracker]
  J --> L[TestGenerationReporter]
```

Refinement loop (sequence)
```mermaid
sequenceDiagram
  participant GenS as TestGenerationService
  participant Py as Pytest Runner
  participant FP as Failure Parser
  participant PB as Payload Builder
  participant LLM as LLMClient (refine)
  participant RM as Refine Manager
  participant W as Safe Apply (tests/ only)

  GenS->>Py: run pytest
  Py-->>GenS: exit_code, stdout/stderr, junit
  alt failures
    GenS->>FP: parse failures
    FP-->>GenS: normalized failures
    GenS->>PB: build payload (env, repo meta, tests, failures)
    PB-->>GenS: payload + prompt
    GenS->>RM: run_refinement_cycle(payload)
    loop attempts <= max_retries
      RM->>LLM: refine_tests(payload, prompt)
      LLM-->>RM: updated_files[]
      alt any updates
        RM->>W: apply updates (guard: tests/)
      end
      RM->>Py: re-run pytest
      Py-->>RM: exit_code
      alt passed or no-change
        RM-->>GenS: outcome
      end
    end
  else no failures
    GenS-->>GenS: skip refinement
  end
```

CLI

Modes
- generate (default): create missing tests
- analyze: show files/elements to generate
- coverage: print coverage summary
- status: recent generation log
- init-config: write a sample .testgen.yml
- cost: cost usage summary (if cost manager enabled)
- debug-state | sync-state | reset-state: state management

Common flags
--verbose, -v; --quiet, -q; --force; --dry-run; --batch-size N; --streaming
--runner-mode [python-module|pytest-path|custom], --pytest-path PATH, --pytest-arg ARG
--auto-run, --refine-enable, --retries N, --max-total-minutes N
--merge-strategy [append|ast-merge], --merge-formatter [none|black], --merge-dry-run

Minimal configuration (.testgen.yml)
```yaml
test_generation:
  coverage:
    minimum_line_coverage: 80
    runner:
      mode: python-module   # or pytest-path
      python: python
      cwd: .
    pytest_args: ["--tb=short"]
  generation:
    merge:
      strategy: append      # or ast-merge
      formatter: none       # or black
    test_runner:
      enable: false         # set true to auto-run pytest after generation
    refine:
      enable: false         # set true to enable bounded refinement loop
exclude_dirs: [".venv", "venv", "node_modules", "site-packages", ".git"]
```

Provider credentials
```bash
# Claude
export CLAUDE_API_KEY=...

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT=...  # https://<resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=...

# AWS Bedrock
export AWS_BEDROCK_ROLE_ARN=arn:aws:iam::...:role/...
export AWS_BEDROCK_INFERENCE_PROFILE=...
```

Merging behavior
- append: append new tests to existing test file (default)
- ast-merge: parse both sides and structurally merge; optional Black formatting
- dry-run: compute unified diff without writing

Coverage model
- Preferred: run pytest with coverage.py, parse .coverage, map executed/missing lines and covered/uncovered functions
- Fallback: ASTCoverageAnalyzer estimates coverage and untested functions when pytest/coverage are unavailable

State and outputs
- .testgen_state.json: tested elements, coverage history, generation log
- .testgen_report.json: summary of a run
- .artifacts/coverage and .artifacts/refine/<run_id>: runner outputs and refinement traces

Safety and limits
- Generated test content is validated: syntax check, size limits, and blocks dangerous patterns (eval/exec/subprocess, write modes)
- Refinement applies updates only under tests/

Troubleshooting
- No coverage file: ensure pytest and coverage are importable in the active venv; try --runner-mode=pytest-path --pytest-path "$(which pytest)"
- Empty/partial LLM JSON: reduce --batch-size or use --streaming
- Imports fail during tests: add src/ to PYTHONPATH via config test_generation.coverage.env.append_pythonpath: ['src']

Development (with uv)
```bash
uv pip install -e .[dev]
uv run pytest
uvx ruff check src tests
uvx black src tests
```

License
MIT

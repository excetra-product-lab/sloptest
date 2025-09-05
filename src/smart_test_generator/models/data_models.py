"""Data models for test generation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum


@dataclass
class FileInfo:
    """Information about a Python file."""
    filepath: str
    filename: str
    content: str


@dataclass
class TestableElement:
    """Represents a testable element in code."""
    name: str
    type: str  # 'function', 'method', 'class'
    filepath: str
    line_number: int
    signature: str
    docstring: Optional[str] = None
    complexity: int = 1
    dependencies: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)


@dataclass
class TestCoverage:
    """Coverage information for a file."""
    filepath: str
    line_coverage: float
    branch_coverage: float
    missing_lines: List[int]
    covered_functions: Set[str]
    uncovered_functions: Set[str]


@dataclass
class TestGenerationPlan:
    """Plan for generating tests."""
    source_file: str
    existing_test_files: List[str]
    elements_to_test: List[TestableElement]
    coverage_before: Optional[TestCoverage]
    estimated_coverage_after: float
    # NEW: Quality-aware fields
    mutation_score_target: float = 80.0
    quality_score_target: float = 95.0
    weak_mutation_spots: List['WeakSpot'] = field(default_factory=list)


@dataclass
class TestGenerationState:
    """State tracking for test generation."""
    timestamp: str
    tested_elements: Dict[str, List[str]]  # filepath -> list of tested element names
    coverage_history: Dict[str, List[float]]  # filepath -> list of coverage percentages
    generation_log: List[Dict[str, Any]]  # log of generation activities


# === NEW: Quality Scoring Models ===

class QualityDimension(Enum):
    """Different dimensions of test quality."""
    EDGE_CASE_COVERAGE = "edge_case_coverage"
    ASSERTION_STRENGTH = "assertion_strength"
    MAINTAINABILITY = "maintainability"
    BUG_DETECTION_POTENTIAL = "bug_detection_potential"
    READABILITY = "readability"
    INDEPENDENCE = "independence"


@dataclass
class QualityScore:
    """Score for a specific quality dimension."""
    dimension: QualityDimension
    score: float  # 0-100
    max_score: float = 100.0
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class TestQualityReport:
    """Comprehensive quality report for a test or test suite."""
    test_file: str
    overall_score: float  # Weighted average of all dimensions
    dimension_scores: Dict[QualityDimension, QualityScore] = field(default_factory=dict)
    
    # Detailed analysis
    edge_cases_found: List[str] = field(default_factory=list)
    edge_cases_missing: List[str] = field(default_factory=list)
    weak_assertions: List[Dict[str, Any]] = field(default_factory=list)
    maintainability_issues: List[str] = field(default_factory=list)
    
    # Improvement suggestions
    improvement_suggestions: List[str] = field(default_factory=list)
    priority_fixes: List[str] = field(default_factory=list)
    
    def get_score(self, dimension: QualityDimension) -> float:
        """Get score for a specific dimension."""
        return self.dimension_scores.get(dimension, QualityScore(dimension, 0.0)).score


# === NEW: Mutation Testing Models ===

class MutationType(Enum):
    """Types of mutations that can be applied to code."""
    ARITHMETIC_OPERATOR = "arithmetic_operator"  # +, -, *, /
    COMPARISON_OPERATOR = "comparison_operator"  # ==, !=, <, >, <=, >=
    LOGICAL_OPERATOR = "logical_operator"        # and, or, not
    CONSTANT_VALUE = "constant_value"            # numbers, strings, booleans
    BOUNDARY_VALUE = "boundary_value"            # off-by-one errors
    CONDITIONAL = "conditional"                  # if/else logic
    LOOP_MUTATION = "loop_mutation"              # loop conditions
    EXCEPTION_HANDLING = "exception_handling"   # try/catch modifications
    METHOD_CALL = "method_call"                  # method invocation changes
    # Modern Python mutation types
    TYPE_HINT = "type_hint"                      # Optional[T] → T, Union mutations
    ASYNC_AWAIT = "async_await"                  # async/await mutations
    DATACLASS = "dataclass"                      # @dataclass mutations


@dataclass
class Mutant:
    """Represents a single mutation applied to source code."""
    id: str
    mutation_type: MutationType
    original_code: str
    mutated_code: str
    line_number: int
    column_start: int
    column_end: int
    description: str
    severity: str = "medium"  # low, medium, high, critical
    
    # Language-specific information
    language: str = "python"
    ast_node_type: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationResult:
    """Result of running tests against a mutant."""
    mutant: Mutant
    killed: bool  # True if tests detected the mutation (test failed)
    execution_time: float
    failing_tests: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    exit_code: int = 0


@dataclass
class WeakSpot:
    """Represents a weak spot in test coverage found through mutation testing."""
    location: str  # file:line:column
    mutation_type: MutationType
    surviving_mutants: List[Mutant]
    description: str
    suggested_tests: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    
    # Context for test generation
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    surrounding_code: str = ""


@dataclass
class MutationScore:
    """Overall mutation testing results."""
    source_file: str
    test_files: List[str]
    
    # Core metrics
    total_mutants: int
    killed_mutants: int
    surviving_mutants: int
    timeout_mutants: int
    error_mutants: int
    
    # Calculated scores
    mutation_score: float  # (killed / total) * 100
    mutation_score_effective: float  # (killed / (total - timeout - error)) * 100
    
    # Detailed results
    mutant_results: List[MutationResult] = field(default_factory=list)
    weak_spots: List[WeakSpot] = field(default_factory=list)
    mutation_distribution: Dict[MutationType, int] = field(default_factory=dict)
    
    # Performance metrics
    total_execution_time: float = 0.0
    average_test_time: float = 0.0
    
    def get_surviving_mutants(self) -> List[Mutant]:
        """Get all mutants that survived (weren't killed by tests)."""
        return [result.mutant for result in self.mutant_results if not result.killed]
    
    def get_weak_spots_by_severity(self, severity: str) -> List[WeakSpot]:
        """Get weak spots filtered by severity level."""
        return [spot for spot in self.weak_spots if spot.severity == severity]


@dataclass
class QualityAndMutationReport:
    """Combined report of test quality and mutation testing results."""
    source_file: str
    test_files: List[str]
    
    # Quality analysis
    quality_report: TestQualityReport
    
    # Mutation analysis
    mutation_score: MutationScore
    
    # Combined insights
    overall_rating: str  # excellent, good, fair, poor
    confidence_level: float  # How confident we are that tests will catch bugs
    recommended_actions: List[str] = field(default_factory=list)
    
    # Prioritized improvement plan
    high_priority_fixes: List[Dict[str, Any]] = field(default_factory=list)
    medium_priority_fixes: List[Dict[str, Any]] = field(default_factory=list)
    low_priority_fixes: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_combined_score(self) -> float:
        """Calculate a combined score from quality and mutation metrics."""
        quality_weight = 0.4
        mutation_weight = 0.6
        
        return (
            self.quality_report.overall_score * quality_weight +
            self.mutation_score.mutation_score * mutation_weight
        )

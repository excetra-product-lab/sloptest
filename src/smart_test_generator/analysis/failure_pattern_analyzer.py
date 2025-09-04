"""Failure pattern analysis for intelligent test refinement.

This module analyzes test failure patterns to provide categorized insights and
automated fix suggestions for common testing issues.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import Counter, defaultdict
import json

from smart_test_generator.analysis.coverage.failure_parser import FailureRecord, ParsedFailures

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of test failure patterns."""
    ASSERTION_ERROR = "assertion_error"
    IMPORT_ERROR = "import_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    NAME_ERROR = "name_error"
    FILE_NOT_FOUND = "file_not_found"
    TIMEOUT_ERROR = "timeout_error"
    FIXTURE_ERROR = "fixture_error"
    MOCK_ERROR = "mock_error"
    SETUP_TEARDOWN_ERROR = "setup_teardown_error"
    DEPENDENCY_ERROR = "dependency_error"
    PARAMETRIZATION_ERROR = "parametrization_error"
    UNKNOWN = "unknown"


@dataclass
class FailurePattern:
    """Represents a detected failure pattern."""
    category: FailureCategory
    pattern: str  # Regex pattern that matches this failure type
    description: str
    frequency: int = 0
    confidence_score: float = 0.0  # How confident we are in this classification
    

@dataclass
class FixSuggestion:
    """Represents a suggested fix for a failure pattern."""
    category: FailureCategory
    title: str
    description: str
    code_example: Optional[str] = None
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    automated: bool = False  # Whether this fix can be applied automatically
    

@dataclass
class FailureAnalysis:
    """Result of analyzing failure patterns."""
    total_failures: int
    categorized_failures: Dict[FailureCategory, List[FailureRecord]] = field(default_factory=dict)
    pattern_frequencies: Dict[FailureCategory, int] = field(default_factory=dict)
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    trending_patterns: List[Tuple[FailureCategory, float]] = field(default_factory=list)


@dataclass
class FailureHistory:
    """Historical tracking of failure patterns."""
    failure_counts: Dict[str, int] = field(default_factory=Counter)  # nodeid -> count
    category_trends: Dict[FailureCategory, List[int]] = field(default_factory=lambda: defaultdict(list))
    resolution_success: Dict[FailureCategory, float] = field(default_factory=dict)  # Success rate
    last_seen: Dict[str, str] = field(default_factory=dict)  # nodeid -> timestamp
    

class FailurePatternAnalyzer:
    """Analyzes test failure patterns and provides intelligent fix suggestions."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / ".smart_test_generator" / "failure_history.json"
        self.failure_patterns = self._initialize_patterns()
        self.fix_suggestions_db = self._initialize_fix_suggestions()
        self.history = self._load_history()
    
    def analyze_failures(self, failures: ParsedFailures) -> FailureAnalysis:
        """Analyze failure patterns and provide categorized insights.
        
        Args:
            failures: Parsed failure records
            
        Returns:
            FailureAnalysis with categorized failures and suggestions
        """
        analysis = FailureAnalysis(total_failures=failures.total)
        
        # Categorize each failure
        for failure in failures.failures:
            category, confidence = self._categorize_failure(failure)
            
            if category not in analysis.categorized_failures:
                analysis.categorized_failures[category] = []
                analysis.pattern_frequencies[category] = 0
                
            analysis.categorized_failures[category].append(failure)
            analysis.pattern_frequencies[category] += 1
            analysis.confidence_scores[failure.nodeid] = confidence
            
            # Update failure history
            self._update_failure_history(failure, category)
        
        # Generate fix suggestions based on patterns
        analysis.fix_suggestions = self._generate_fix_suggestions(analysis)
        
        # Calculate trending patterns
        analysis.trending_patterns = self._calculate_trending_patterns(analysis)
        
        # Save updated history
        self._save_history()
        
        return analysis
    
    def get_failure_trends(self, days_back: int = 30) -> Dict[FailureCategory, List[int]]:
        """Get failure trends over time.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary mapping categories to trend data
        """
        return dict(self.history.category_trends)
    
    def get_success_rates(self) -> Dict[FailureCategory, float]:
        """Get resolution success rates by category.
        
        Returns:
            Dictionary mapping categories to success rates (0.0-1.0)
        """
        return dict(self.history.resolution_success)
    
    def mark_resolution_success(self, nodeid: str, category: FailureCategory, success: bool):
        """Mark whether a fix attempt was successful.
        
        Args:
            nodeid: Test node identifier
            category: Failure category
            success: Whether the fix was successful
        """
        if category not in self.history.resolution_success:
            self.history.resolution_success[category] = 0.0
        
        # Update success rate using exponential moving average
        current_rate = self.history.resolution_success[category]
        new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        self.history.resolution_success[category] = new_rate
        
        # Save updated history
        self._save_history()
    
    def _categorize_failure(self, failure: FailureRecord) -> Tuple[FailureCategory, float]:
        """Categorize a single failure record.
        
        Args:
            failure: Failure record to categorize
            
        Returns:
            Tuple of (category, confidence_score)
        """
        message = failure.message.lower()
        assertion_diff = (failure.assertion_diff or "").lower()
        combined_text = f"{message} {assertion_diff}"
        
        best_match = FailureCategory.UNKNOWN
        best_confidence = 0.0
        
        for category, patterns in self.failure_patterns.items():
            for pattern in patterns:
                match_count = len(re.findall(pattern.pattern, combined_text, re.IGNORECASE))
                if match_count > 0:
                    # Calculate confidence based on pattern specificity and match quality
                    confidence = min(0.95, 0.3 + (match_count * 0.2) + (len(pattern.pattern) / 100))
                    
                    if confidence > best_confidence:
                        best_match = category
                        best_confidence = confidence
        
        return best_match, best_confidence
    
    def _generate_fix_suggestions(self, analysis: FailureAnalysis) -> List[FixSuggestion]:
        """Generate fix suggestions based on failure analysis.
        
        Args:
            analysis: Current failure analysis
            
        Returns:
            List of prioritized fix suggestions
        """
        suggestions = []
        
        for category, failures in analysis.categorized_failures.items():
            if category == FailureCategory.UNKNOWN:
                continue
                
            # Get base suggestions for this category
            base_suggestions = self.fix_suggestions_db.get(category, [])
            
            for base_suggestion in base_suggestions:
                # Customize suggestion based on specific failures
                customized = self._customize_suggestion(base_suggestion, failures)
                if customized:
                    suggestions.append(customized)
        
        # Sort by priority and frequency
        suggestions.sort(key=lambda s: (s.priority, -analysis.pattern_frequencies.get(s.category, 0)))
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def _customize_suggestion(self, base_suggestion: FixSuggestion, 
                            failures: List[FailureRecord]) -> Optional[FixSuggestion]:
        """Customize a fix suggestion based on specific failure instances.
        
        Args:
            base_suggestion: Base fix suggestion template
            failures: List of failures in this category
            
        Returns:
            Customized fix suggestion or None
        """
        if not failures:
            return None
        
        # Analyze common patterns in the failures
        common_files = Counter(f.file for f in failures if f.file)
        common_functions = self._extract_function_names(failures)
        
        # Customize the description with specific details
        customized_desc = base_suggestion.description
        
        if common_files:
            most_common_file = common_files.most_common(1)[0][0]
            customized_desc += f"\n\nMost affected file: {most_common_file}"
        
        if common_functions:
            most_common_func = common_functions.most_common(1)[0][0]
            customized_desc += f"\nMost affected function: {most_common_func}"
        
        # Adjust priority based on frequency and history
        priority = base_suggestion.priority
        failure_count = len(failures)
        
        if failure_count > 5:
            priority = max(1, priority - 1)  # Increase priority for frequent failures
        
        return FixSuggestion(
            category=base_suggestion.category,
            title=base_suggestion.title,
            description=customized_desc,
            code_example=base_suggestion.code_example,
            priority=priority,
            automated=base_suggestion.automated
        )
    
    def _calculate_trending_patterns(self, analysis: FailureAnalysis) -> List[Tuple[FailureCategory, float]]:
        """Calculate which failure patterns are trending.
        
        Args:
            analysis: Current failure analysis
            
        Returns:
            List of (category, trend_score) sorted by trend strength
        """
        trends = []
        
        for category, current_count in analysis.pattern_frequencies.items():
            if category == FailureCategory.UNKNOWN:
                continue
                
            # Get historical trend for this category
            historical = self.history.category_trends.get(category, [])
            
            if len(historical) < 2:
                trend_score = 1.0  # New pattern
            else:
                # Calculate trend as recent average vs older average
                recent_avg = sum(historical[-3:]) / min(3, len(historical))
                older_avg = sum(historical[:-3]) / max(1, len(historical) - 3)
                
                if older_avg == 0:
                    trend_score = 1.0
                else:
                    trend_score = recent_avg / older_avg
            
            trends.append((category, trend_score))
        
        # Sort by trend strength (descending)
        trends.sort(key=lambda x: x[1], reverse=True)
        
        return trends[:5]  # Top 5 trending patterns
    
    def _update_failure_history(self, failure: FailureRecord, category: FailureCategory):
        """Update failure history with new failure information."""
        # Update failure count
        self.history.failure_counts[failure.nodeid] += 1
        
        # Update category trend
        if category not in self.history.category_trends:
            self.history.category_trends[category] = []
        
        # Keep last 30 data points for trends
        trend_data = self.history.category_trends[category]
        if len(trend_data) >= 30:
            trend_data.pop(0)
        trend_data.append(1)
        
        # Update last seen timestamp
        import time
        self.history.last_seen[failure.nodeid] = str(int(time.time()))
    
    def _extract_function_names(self, failures: List[FailureRecord]) -> Counter:
        """Extract function names from failure nodeids."""
        functions = []
        for failure in failures:
            # Extract function name from nodeid (e.g., "file::class::test_func")
            parts = failure.nodeid.split("::")
            if len(parts) >= 2:
                functions.append(parts[-1])  # Last part is usually the function name
        
        return Counter(functions)
    
    def _load_history(self) -> FailureHistory:
        """Load failure history from disk."""
        if not self.history_file.exists():
            return FailureHistory()
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history = FailureHistory()
            history.failure_counts = Counter(data.get('failure_counts', {}))
            history.last_seen = data.get('last_seen', {})
            history.resolution_success = {
                FailureCategory(k): v for k, v in data.get('resolution_success', {}).items()
            }
            
            # Convert category trends
            for category_str, trend_list in data.get('category_trends', {}).items():
                try:
                    category = FailureCategory(category_str)
                    history.category_trends[category] = trend_list
                except ValueError:
                    continue  # Skip unknown categories
            
            return history
            
        except Exception as e:
            logger.warning(f"Failed to load failure history: {e}")
            return FailureHistory()
    
    def _save_history(self):
        """Save failure history to disk."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'failure_counts': dict(self.history.failure_counts),
                'last_seen': self.history.last_seen,
                'resolution_success': {
                    k.value: v for k, v in self.history.resolution_success.items()
                },
                'category_trends': {
                    k.value: v for k, v in self.history.category_trends.items()
                }
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save failure history: {e}")
    
    def _initialize_patterns(self) -> Dict[FailureCategory, List[FailurePattern]]:
        """Initialize failure pattern definitions."""
        return {
            FailureCategory.ASSERTION_ERROR: [
                FailurePattern(
                    FailureCategory.ASSERTION_ERROR,
                    r"assert(ionerror)?.*(?:==|!=|<|>|in|not in)",
                    "Assertion comparison failure"
                ),
                FailurePattern(
                    FailureCategory.ASSERTION_ERROR,
                    r"expected.*(?:but.*was|actual|got)",
                    "Expected vs actual value mismatch"
                ),
            ],
            
            FailureCategory.IMPORT_ERROR: [
                FailurePattern(
                    FailureCategory.IMPORT_ERROR,
                    r"importerror|modulenotfounderror|no module named",
                    "Module import failure"
                ),
                FailurePattern(
                    FailureCategory.IMPORT_ERROR,
                    r"cannot import name|importerror: cannot import",
                    "Specific import name failure"
                ),
            ],
            
            FailureCategory.ATTRIBUTE_ERROR: [
                FailurePattern(
                    FailureCategory.ATTRIBUTE_ERROR,
                    r"attributeerror.*has no attribute",
                    "Object missing expected attribute"
                ),
                FailurePattern(
                    FailureCategory.ATTRIBUTE_ERROR,
                    r"'nonetype' object has no attribute",
                    "None object attribute access"
                ),
            ],
            
            FailureCategory.TYPE_ERROR: [
                FailurePattern(
                    FailureCategory.TYPE_ERROR,
                    r"typeerror.*takes.*positional argument",
                    "Function argument count mismatch"
                ),
                FailurePattern(
                    FailureCategory.TYPE_ERROR,
                    r"typeerror.*unsupported operand type",
                    "Type operation incompatibility"
                ),
            ],
            
            FailureCategory.FIXTURE_ERROR: [
                FailurePattern(
                    FailureCategory.FIXTURE_ERROR,
                    r"fixture.*not found|fixture.*error",
                    "Pytest fixture issues"
                ),
                FailurePattern(
                    FailureCategory.FIXTURE_ERROR,
                    r"teardown|setup.*failed",
                    "Test setup or teardown failure"
                ),
            ],
            
            FailureCategory.MOCK_ERROR: [
                FailurePattern(
                    FailureCategory.MOCK_ERROR,
                    r"mock.*object|magicmock|patch",
                    "Mock object related issues"
                ),
                FailurePattern(
                    FailureCategory.MOCK_ERROR,
                    r"assert_called|call_count",
                    "Mock assertion failures"
                ),
            ],
        }
    
    def _initialize_fix_suggestions(self) -> Dict[FailureCategory, List[FixSuggestion]]:
        """Initialize fix suggestion database."""
        return {
            FailureCategory.ASSERTION_ERROR: [
                FixSuggestion(
                    FailureCategory.ASSERTION_ERROR,
                    "Review assertion logic",
                    "Check if the assertion is testing the right condition. Consider using more specific assertion methods.",
                    code_example="# Instead of: assert result == expected\n# Use: assert result == expected, f'Expected {expected}, got {result}'",
                    priority=1
                ),
                FixSuggestion(
                    FailureCategory.ASSERTION_ERROR,
                    "Add assertion messages",
                    "Add descriptive messages to assertions for better debugging.",
                    code_example="assert condition, 'Detailed description of what should be true'",
                    priority=2
                ),
            ],
            
            FailureCategory.IMPORT_ERROR: [
                FixSuggestion(
                    FailureCategory.IMPORT_ERROR,
                    "Check dependencies",
                    "Verify all required packages are installed and importable.",
                    priority=1,
                    automated=True
                ),
                FixSuggestion(
                    FailureCategory.IMPORT_ERROR,
                    "Fix import paths",
                    "Check if import paths are correct and modules exist in the expected locations.",
                    priority=2
                ),
            ],
            
            FailureCategory.ATTRIBUTE_ERROR: [
                FixSuggestion(
                    FailureCategory.ATTRIBUTE_ERROR,
                    "Check object initialization",
                    "Ensure objects are properly initialized before accessing attributes.",
                    priority=1
                ),
                FixSuggestion(
                    FailureCategory.ATTRIBUTE_ERROR,
                    "Add None checks",
                    "Add checks for None values before attribute access.",
                    code_example="if obj is not None:\n    value = obj.attribute",
                    priority=2,
                    automated=True
                ),
            ],
            
            FailureCategory.FIXTURE_ERROR: [
                FixSuggestion(
                    FailureCategory.FIXTURE_ERROR,
                    "Check fixture definitions",
                    "Ensure fixtures are properly defined and accessible in the test scope.",
                    priority=1
                ),
                FixSuggestion(
                    FailureCategory.FIXTURE_ERROR,
                    "Review fixture dependencies",
                    "Check if fixture dependencies are correctly specified and available.",
                    priority=2
                ),
            ],
            
            FailureCategory.MOCK_ERROR: [
                FixSuggestion(
                    FailureCategory.MOCK_ERROR,
                    "Review mock usage",
                    "Check if mocks are being used correctly and expectations match actual calls.",
                    priority=1
                ),
                FixSuggestion(
                    FailureCategory.MOCK_ERROR,
                    "Fix mock assertions",
                    "Ensure mock assertion methods match the actual call patterns.",
                    code_example="mock_obj.assert_called_once_with(expected_args)",
                    priority=2
                ),
            ],
        }


def create_failure_analyzer(project_root: Path) -> FailurePatternAnalyzer:
    """Factory function to create a FailurePatternAnalyzer instance."""
    return FailurePatternAnalyzer(project_root)

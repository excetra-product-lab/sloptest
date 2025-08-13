"""Track test generation state based on coverage."""

import os
import json
import logging
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from dataclasses import asdict

from smart_test_generator.models.data_models import TestGenerationState, TestCoverage
from smart_test_generator.config import Config

logger = logging.getLogger(__name__)


class TestGenerationTracker:
    """Track test generation state based on coverage, not file changes."""

    def __init__(self, state_file: str = ".testgen_state.json"):
        self.state_file = state_file
        # Avoid polluting tests with an existing global state file
        self.state = self._load_state() if os.path.exists(self.state_file) else TestGenerationState(
            timestamp=datetime.now().isoformat(),
            tested_elements={},
            coverage_history={},
            generation_log=[]
        )

    def _load_state(self) -> TestGenerationState:
        """Load previous test generation state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return TestGenerationState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return TestGenerationState(
            timestamp=datetime.now().isoformat(),
            tested_elements={},
            coverage_history={},
            generation_log=[]
        )

    def save_state(self):
        """Save current state to file."""
        try:
            self.state.timestamp = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def should_generate_tests(self, filepath: str, current_coverage: Optional[TestCoverage],
                            config: Config, force: bool = False) -> Tuple[bool, str]:
        """Determine if tests should be generated for a file and why."""
        if force:
            return True, "Force flag set"

        # New file - only considered new if not in state AND no coverage provided
        if filepath not in self.state.tested_elements and not current_coverage:
            return True, "New file detected"

        # No coverage data - check if tests exist before defaulting to generation
        if not current_coverage:
            # Check if we have any tested elements recorded for this file
            if filepath in self.state.tested_elements and self.state.tested_elements[filepath]:
                # We have recorded tested elements, so tests likely exist
                # Only regenerate if we haven't seen this file recently
                if filepath not in self.state.coverage_history:
                    return False, "Tests exist but no recent coverage data - skipping to avoid regeneration"
                return False, "Tested elements exist, no coverage data but avoiding regeneration"
            else:
                # No tested elements recorded and no coverage data
                return True, "No coverage data available and no tested elements recorded"

        # Coverage dropped significantly (prioritize this reason when applicable)
        if filepath in self.state.coverage_history:
            history = self.state.coverage_history[filepath]
            if history and len(history) > 0:
                last_coverage = history[-1]
                coverage_drop = last_coverage - current_coverage.line_coverage
                if coverage_drop > 10:  # Coverage dropped by more than 10%
                    return True, f"Coverage dropped by {coverage_drop:.1f}%"

        # Coverage below minimum threshold
        min_coverage = config.get('test_generation.coverage.minimum_line_coverage', 80)
        if current_coverage.line_coverage < min_coverage:
            return True, f"Coverage ({current_coverage.line_coverage:.1f}%) below minimum ({min_coverage}%)"

        # Check for untested elements
        if current_coverage.uncovered_functions:
            return True, f"{len(current_coverage.uncovered_functions)} untested functions found"

        return False, "Adequate test coverage exists"

    def update_coverage(self, filepath: str, coverage: float):
        """Update coverage history for a file."""
        if filepath not in self.state.coverage_history:
            self.state.coverage_history[filepath] = []
        self.state.coverage_history[filepath].append(coverage)

        # Keep only last 10 coverage values
        if len(self.state.coverage_history[filepath]) > 10:
            self.state.coverage_history[filepath] = self.state.coverage_history[filepath][-10:]

    def record_generation(self, filepath: str, elements_generated: List[str],
                         coverage_before: float, coverage_after: float, reason: str):
        """Record test generation activity."""
        self.state.generation_log.append({
            "timestamp": datetime.now().isoformat(),
            "filepath": filepath,
            "reason": reason,
            "elements_generated": len(elements_generated),
            "elements": elements_generated[:10],  # Store first 10 for reference
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "improvement": coverage_after - coverage_before
        })

        # Update tested elements
        if filepath not in self.state.tested_elements:
            self.state.tested_elements[filepath] = []
        # Deduplicate while preserving order
        existing = self.state.tested_elements[filepath]
        for elem in elements_generated:
            if elem not in existing:
                existing.append(elem)

        # Keep only last 100 log entries
        if len(self.state.generation_log) > 100:
            self.state.generation_log = self.state.generation_log[-100:]

    def reset_state(self):
        """Reset the test generation state (useful when state becomes inconsistent)."""
        logger.info("Resetting test generation state")
        self.state = TestGenerationState(
            timestamp=datetime.now().isoformat(),
            tested_elements={},
            coverage_history={},
            generation_log=[]
        )
        self.save_state()

    def force_mark_as_tested(self, filepath: str, elements: List[str], reason: str = "Manual override"):
        """Manually mark elements as tested for a file."""
        logger.info(f"Manually marking {len(elements)} elements as tested for {filepath}")
        
        if filepath not in self.state.tested_elements:
            self.state.tested_elements[filepath] = []
        
        # Add elements to tested list, preserving order and uniqueness
        existing = self.state.tested_elements[filepath]
        for elem in elements:
            if elem not in existing:
                existing.append(elem)
        
        # Log this action
        self.state.generation_log.append({
            "timestamp": datetime.now().isoformat(),
            "filepath": filepath,
            "reason": reason,
            "elements_generated": len(elements),
            "elements": elements[:10],  # Store first 10 for reference
            "coverage_before": 0,
            "coverage_after": 100,  # Assume good coverage
            "improvement": 100
        })
        
        self.save_state()

    def get_state_summary(self) -> Dict:
        """Get a summary of the current state for debugging."""
        return {
            "timestamp": self.state.timestamp,
            "files_with_tests": len(self.state.tested_elements),
            "total_tested_elements": sum(len(elements) for elements in self.state.tested_elements.values()),
            "files_with_coverage_history": len(self.state.coverage_history),
            "generation_log_entries": len(self.state.generation_log),
            "tested_files": list(self.state.tested_elements.keys())
        }

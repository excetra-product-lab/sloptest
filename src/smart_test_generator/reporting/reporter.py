"""Test generation reporting."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TestGenerationReporter:
    """Generate reports about test generation."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.report_file = self.project_root / ".testgen_report.json"

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Create detailed report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "files_processed": results.get("files_processed", 0),
                "tests_generated": results.get("tests_generated", 0),
                "coverage_before": results.get("coverage_before", 0),
                "coverage_after": results.get("coverage_after", 0),
                "coverage_improvement": round(results.get("coverage_after", 0) - results.get("coverage_before", 0), 1)
            },
            "details": results.get("details", [])
        }

        # Save report
        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        summary = f"""
Test Generation Report
======================
Generated: {report['timestamp']}

Summary:
- Files processed: {report['summary']['files_processed']}
- Tests generated: {report['summary']['tests_generated']}
- Coverage before: {report['summary']['coverage_before']:.1f}%
- Coverage after: {report['summary']['coverage_after']:.1f}%
- Improvement: +{report['summary']['coverage_improvement']:.1f}%

Report saved to: {self.report_file}
"""
        return summary

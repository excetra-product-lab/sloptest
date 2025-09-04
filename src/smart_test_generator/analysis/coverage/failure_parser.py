"""Parse pytest failures from JUnit XML or stdout/stderr into a normalized structure."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from xml.etree import ElementTree as ET


@dataclass
class FailureRecord:
    nodeid: str
    file: str
    line: Optional[int]
    message: str
    assertion_diff: Optional[str]
    captured_stdout: Optional[str]
    captured_stderr: Optional[str]
    duration: Optional[float]


@dataclass
class ParsedFailures:
    total: int
    failures: List[FailureRecord]


def parse_junit_xml(junit_path: Path) -> ParsedFailures:
    failures: List[FailureRecord] = []
    try:
        tree = ET.parse(str(junit_path))
        root = tree.getroot()
        # JUnit schema variations: testsuite/testcase
        for case in root.iter('testcase'):
            classname = case.attrib.get('classname', '')
            name = case.attrib.get('name', '')
            time = case.attrib.get('time')
            duration = float(time) if time else None
            nodeid = f"{classname}::{name}" if classname else name
            file_attr = case.attrib.get('file') or classname.replace('.', '/') + '.py'
            line_attr = case.attrib.get('line')
            line = int(line_attr) if line_attr and line_attr.isdigit() else None

            failure_elem = case.find('failure') or case.find('error')
            if failure_elem is not None:
                message = (failure_elem.attrib.get('message') or '').strip()
                text = (failure_elem.text or '').strip()
                failures.append(
                    FailureRecord(
                        nodeid=nodeid,
                        file=file_attr,
                        line=line,
                        message=message or text,
                        assertion_diff=_extract_assertion_diff(text),
                        captured_stdout=_extract_system_out(case),
                        captured_stderr=_extract_system_err(case),
                        duration=duration,
                    )
                )
    except Exception:
        # On parse error, return empty; caller can fallback to stdout parsing
        return ParsedFailures(total=0, failures=[])

    return ParsedFailures(total=len(failures), failures=failures)


def parse_stdout_stderr(stdout: str, stderr: str) -> ParsedFailures:
    """
    Parse pytest stdout/stderr output to extract test failures.
    
    Handles multiple output formats:
    1. Main test run format: "path::test FAILED/ERROR"
    2. Short summary format: "FAILED/ERROR path::test - message"
    3. Line format: "path:line: ErrorType: message"
    
    Args:
        stdout: Standard output from pytest run
        stderr: Standard error from pytest run
        
    Returns:
        ParsedFailures object containing detected failures
    """
    if not stdout and not stderr:
        return ParsedFailures(total=0, failures=[])
        
    text = stdout + "\n" + stderr
    failures: List[FailureRecord] = []
    seen_nodeids = set()  # Prevent duplicates
    
    try:
        # Pattern 0: Parallel execution format (pytest-xdist) - [gwN] prefix
        parallel_pattern = r"^\[gw\d+\]\s+(?P<status>FAILED|ERROR)\s+(?P<path>[^\s]+\.py(?:::[\w\[\]:,\-\.\{\}\"'\/\?=]*)*)"
        for match in re.finditer(parallel_pattern, text, re.MULTILINE):
            nodeid = match.group('path')
            if nodeid in seen_nodeids:
                continue
            seen_nodeids.add(nodeid)
            
            file_path = nodeid.split('::')[0]
            status = match.group('status')
            
            failures.append(
                FailureRecord(
                    nodeid=nodeid,
                    file=file_path,
                    line=None,
                    message=f"Test {status.lower()} (parallel execution)",
                    assertion_diff=None,
                    captured_stdout=None,
                    captured_stderr=None,
                    duration=None,
                )
            )
        
        # Pattern 1: Main test run output (path::test FAILED/ERROR)
        # Enhanced regex for complex parametrized tests with JSON, URLs, special chars
        main_pattern = r"^(?P<path>[^\s]+\.py(?:::[\w\[\]:,\-\.\{\}\"'\/\?=]*)*)\s+(?P<status>FAILED|ERROR)"
        for match in re.finditer(main_pattern, text, re.MULTILINE):
            nodeid = match.group('path')
            if nodeid in seen_nodeids:
                continue
            seen_nodeids.add(nodeid)
            
            file_path = nodeid.split('::')[0]
            status = match.group('status')
            
            # Try to capture assertion context following this line
            start = match.end()
            snippet = text[start:start + 500]
            
            failures.append(
                FailureRecord(
                    nodeid=nodeid,
                    file=file_path,
                    line=None,
                    message=f"Test {status.lower()}",
                    assertion_diff=_extract_assertion_diff(snippet),
                    captured_stdout=None,
                    captured_stderr=None,
                    duration=None,
                )
            )
        
        # Pattern 2: Short summary format (FAILED/ERROR path::test [- message])
        # Enhanced for complex parametrized tests and optional messages
        summary_pattern = r"^(?P<status>FAILED|ERROR)\s+(?P<path>[^\s]+\.py(?:::[\w\[\]:,\-\.\{\}\"'\/\?=]*)*)(?:\s+-\s+(?P<message>.*))?"
        for match in re.finditer(summary_pattern, text, re.MULTILINE):
            nodeid = match.group('path')
            if nodeid in seen_nodeids:
                # Update existing failure with better message if available
                message_group = match.group('message')
                if message_group:
                    for failure in failures:
                        if failure.nodeid == nodeid:
                            failure.message = message_group.strip()
                            break
                continue
            seen_nodeids.add(nodeid)
            
            file_path = nodeid.split('::')[0]
            message_group = match.group('message')
            message = message_group.strip() if message_group else f"Test {match.group('status').lower()}"
            
            failures.append(
                FailureRecord(
                    nodeid=nodeid,
                    file=file_path,
                    line=None,
                    message=message,
                    assertion_diff=_extract_assertion_diff(message),
                    captured_stdout=None,
                    captured_stderr=None,
                    duration=None,
                )
            )
        
        # Pattern 3: Line format output (path:line: error_type: message)
        line_pattern = r"^(?P<path>[^\s]+\.py):(?P<line>\d+):\s+(?P<error_type>\w+Error|AssertionError):\s+(?P<message>.*)"
        for match in re.finditer(line_pattern, text, re.MULTILINE):
            file_path = match.group('path')
            line_num = int(match.group('line'))
            error_type = match.group('error_type')
            message = match.group('message').strip()
            
            # Create a synthetic nodeid for line-format failures
            nodeid = f"{file_path}::line_{line_num}"
            if nodeid in seen_nodeids:
                continue
            seen_nodeids.add(nodeid)
            
            failures.append(
                FailureRecord(
                    nodeid=nodeid,
                    file=file_path,
                    line=line_num,
                    message=f"{error_type}: {message}",
                    assertion_diff=_extract_assertion_diff(message),
                    captured_stdout=None,
                    captured_stderr=None,
                    duration=None,
                )
            )
        
        # Pattern 4: Collection errors - handle these carefully
        # These are import/syntax errors that prevent test collection, not actual test failures
        # Format: "ERROR tests/file.py - ImportError: message"
        collection_pattern = r"^(?P<status>ERROR)\s+(?P<path>[^\s]+\.py)\s+-\s+(?P<message>(?:ImportError|SyntaxError|ModuleNotFoundError).*)"
        for match in re.finditer(collection_pattern, text, re.MULTILINE):
            file_path = match.group('path')
            message = match.group('message').strip()
            
            # Use file path as nodeid for collection errors
            nodeid = f"{file_path}::collection_error"
            if nodeid in seen_nodeids:
                continue
            seen_nodeids.add(nodeid)
            
            # Collection errors are different from test failures - they prevent tests from running
            # Mark them clearly so refinement can handle them differently
            failures.append(
                FailureRecord(
                    nodeid=nodeid,
                    file=file_path,
                    line=None,
                    message=f"Collection error: {message}",
                    assertion_diff=None,
                    captured_stdout=None,
                    captured_stderr=None,
                    duration=None,
                )
            )
    
    except Exception as e:
        # If parsing fails, return what we have so far
        # Log the error but don't crash the whole process
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error parsing pytest output: {e}")
    
    return ParsedFailures(total=len(failures), failures=failures)


def write_failures_json(parsed: ParsedFailures, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "failures.json"
    payload: Dict[str, Any] = {
        "total": parsed.total,
        "failures": [asdict(f) for f in parsed.failures],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _extract_system_out(case_elem) -> Optional[str]:
    sysout = case_elem.find('system-out')
    return sysout.text.strip() if sysout is not None and sysout.text else None


def _extract_system_err(case_elem) -> Optional[str]:
    syserr = case_elem.find('system-err')
    return syserr.text.strip() if syserr is not None and syserr.text else None


def _extract_assertion_diff(text: str) -> Optional[str]:
    # Simple heuristic: grab a section starting with 'AssertionError' or 'assert '
    m = re.search(r"(AssertionError[\s\S]{0,400})", text)
    if m:
        return m.group(1)
    m = re.search(r"(assert [\s\S]{0,400})", text)
    if m:
        return m.group(1)
    return None


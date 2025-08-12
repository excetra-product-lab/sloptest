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
    text = stdout + "\n" + stderr
    failures: List[FailureRecord] = []
    # Very lightweight heuristic: lines like path::testname FAILED and capture assertion blocks
    for match in re.finditer(r"^(?P<path>[^\s:]+\.py(::[\w\[\]':]+)+)\s+FAILED", text, re.MULTILINE):
        nodeid = match.group('path')
        file_path = nodeid.split('::')[0]
        # Try to capture a small assertion context following this line
        start = match.end()
        snippet = text[start:start + 500]
        failures.append(
            FailureRecord(
                nodeid=nodeid,
                file=file_path,
                line=None,
                message="Test failed",
                assertion_diff=_extract_assertion_diff(snippet),
                captured_stdout=None,
                captured_stderr=None,
                duration=None,
            )
        )
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


from pathlib import Path
from smart_test_generator.analysis.coverage.failure_parser import (
    parse_stdout_stderr,
    write_failures_json,
    parse_junit_xml,
)


def test_parse_stdout_stderr_extracts_basic_failure(tmp_path: Path):
    stdout = "tests/test_sample.py::test_x FAILED\n"
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.nodeid.startswith("tests/test_sample.py::test_x")
    out = write_failures_json(parsed, tmp_path)
    assert out.exists()


def test_parse_stdout_stderr_handles_error_status():
    """Test parsing ERROR status (not just FAILED)"""
    stdout = "tests/test_sample.py::test_fixture_error ERROR\n"
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.nodeid == "tests/test_sample.py::test_fixture_error"
    assert rec.file == "tests/test_sample.py"


def test_parse_stdout_stderr_handles_parametrized_tests():
    """Test parsing parametrized tests with complex nodeids"""
    stdout = "tests/test_sample.py::test_param[hello-world] FAILED\n"
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.nodeid == "tests/test_sample.py::test_param[hello-world]"
    assert rec.file == "tests/test_sample.py"


def test_parse_stdout_stderr_handles_class_methods():
    """Test parsing test methods in classes"""
    stdout = "tests/test_sample.py::TestClass::test_method FAILED\n"
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.nodeid == "tests/test_sample.py::TestClass::test_method"
    assert rec.file == "tests/test_sample.py"


def test_parse_stdout_stderr_handles_short_test_summary_without_message():
    """Test parsing short test summary format without error messages (common pytest format)"""
    stdout = """=========================== short test summary info ============================
FAILED tests/weather_collector/test_analytics.py::TestWeatherAnalytics::test_get_temperature_stats_with_null_values
FAILED tests/weather_collector/test_api_client.py::TestWeatherAPIClient::test_init_initializes_client_with_config
FAILED tests/weather_collector/test_api_client.py::TestWeatherAPIClient::test_parse_weather_data_missing_weather_array_raises_error
================== 3 failed, 115 passed, 3 warnings in 1.36s ==================="""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 3
    
    # Verify all three failures are captured correctly
    expected_nodeids = [
        "tests/weather_collector/test_analytics.py::TestWeatherAnalytics::test_get_temperature_stats_with_null_values",
        "tests/weather_collector/test_api_client.py::TestWeatherAPIClient::test_init_initializes_client_with_config",
        "tests/weather_collector/test_api_client.py::TestWeatherAPIClient::test_parse_weather_data_missing_weather_array_raises_error"
    ]
    
    actual_nodeids = [f.nodeid for f in parsed.failures]
    for expected in expected_nodeids:
        assert expected in actual_nodeids, f"Expected {expected} to be in {actual_nodeids}"
    
    # Verify messages are generated when no explicit message is provided
    for failure in parsed.failures:
        assert failure.message == "Test failed"
        assert failure.file.endswith('.py')


def test_parse_stdout_stderr_handles_parallel_execution_format():
    """Test parsing pytest-xdist parallel execution output with [gwN] prefixes"""
    stdout = """[gw0] FAILED tests/test_parallel1.py::test_function
[gw1] ERROR tests/test_parallel2.py::test_fixture_error
[gw0] FAILED tests/test_parallel3.py::TestClass::test_method"""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 3
    
    expected_nodeids = [
        "tests/test_parallel1.py::test_function",
        "tests/test_parallel2.py::test_fixture_error", 
        "tests/test_parallel3.py::TestClass::test_method"
    ]
    
    actual_nodeids = [f.nodeid for f in parsed.failures]
    for expected in expected_nodeids:
        assert expected in actual_nodeids, f"Expected {expected} to be in {actual_nodeids}"
    
    # Verify parallel execution is noted in messages
    for failure in parsed.failures:
        assert "(parallel execution)" in failure.message


def test_parse_stdout_stderr_handles_collection_errors():
    """Test parsing collection errors (import/syntax errors) distinctly from test failures"""
    stdout = """ERROR tests/test_broken.py - ImportError: No module named 'missing_module'
FAILED tests/test_normal.py::test_works"""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    # We expect 3 total: regular ERROR match, FAILED match, and collection error match
    assert parsed.total == 3
    
    # Find the collection error and regular test failure
    collection_errors = [f for f in parsed.failures if "collection_error" in f.nodeid]
    test_failures = [f for f in parsed.failures if f.nodeid == "tests/test_normal.py::test_works"]
    regular_errors = [f for f in parsed.failures if f.nodeid == "tests/test_broken.py"]
    
    assert len(collection_errors) == 1, f"Expected 1 collection error, got {len(collection_errors)}"
    assert len(test_failures) == 1, f"Expected 1 test failure, got {len(test_failures)}" 
    assert len(regular_errors) == 1, f"Expected 1 regular error, got {len(regular_errors)}"
    
    # Verify collection error details (most specific)
    coll_error = collection_errors[0]
    assert coll_error.file == "tests/test_broken.py"
    assert "Collection error: ImportError" in coll_error.message
    
    # Verify test failure details  
    test_error = test_failures[0]
    assert test_error.nodeid == "tests/test_normal.py::test_works"
    
    # Collection errors should be identifiable for different handling in refinement
    assert any("Collection error" in f.message for f in parsed.failures)


def test_parse_stdout_stderr_handles_multiple_failures():
    """Test parsing multiple failures in output"""
    stdout = """
temp_test_examples.py::test_simple_assertion_failure FAILED                                          [ 12%]
temp_test_examples.py::test_exception_error FAILED                                                   [ 25%]
temp_test_examples.py::test_parametrized_with_params[hello-world] FAILED                             [ 50%]
temp_error_examples.py::test_using_broken_fixture ERROR                                              [ 50%]
"""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 4
    
    # Check that all different types are captured
    nodeids = [f.nodeid for f in parsed.failures]
    assert "temp_test_examples.py::test_simple_assertion_failure" in nodeids
    assert "temp_test_examples.py::test_exception_error" in nodeids
    assert "temp_test_examples.py::test_parametrized_with_params[hello-world]" in nodeids
    assert "temp_error_examples.py::test_using_broken_fixture" in nodeids


def test_parse_stdout_stderr_handles_short_summary_format():
    """Test parsing failures from the short summary section"""
    stdout = """
========================================= short test summary info ==========================================
FAILED temp_test_examples.py::test_simple_assertion_failure - assert 1 == 2
ERROR temp_error_examples.py::test_using_broken_fixture - RuntimeError: Fixture setup failed
FAILED temp_test_examples.py::test_parametrized_with_params[foo-bar] - AssertionError: assert 'foo' == 'bar'
"""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 3
    
    # Verify failure messages are captured
    messages = [f.message for f in parsed.failures]
    assert any("assert 1 == 2" in msg for msg in messages)
    assert any("RuntimeError: Fixture setup failed" in msg for msg in messages)
    assert any("AssertionError: assert 'foo' == 'bar'" in msg for msg in messages)


def test_parse_stdout_stderr_handles_complex_paths():
    """Test parsing with complex file paths"""
    stdout = "path/to/deep/test_file.py::TestComplexClass::test_nested_method[param-with-dashes] FAILED\n"
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.nodeid == "path/to/deep/test_file.py::TestComplexClass::test_nested_method[param-with-dashes]"
    assert rec.file == "path/to/deep/test_file.py"


def test_parse_stdout_stderr_ignores_non_failures():
    """Test that parsing ignores SKIPPED, XFAIL, and PASSED statuses"""
    stdout = """
test_sample.py::test_passed PASSED
test_sample.py::test_skipped SKIPPED
test_sample.py::test_xfail XFAIL
test_sample.py::test_failed FAILED
test_sample.py::test_error ERROR
"""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 2  # Only FAILED and ERROR should be captured
    
    nodeids = [f.nodeid for f in parsed.failures]
    assert "test_sample.py::test_failed" in nodeids
    assert "test_sample.py::test_error" in nodeids


def test_parse_stdout_stderr_handles_empty_output():
    """Test parsing empty or whitespace-only output"""
    parsed = parse_stdout_stderr("", "")
    assert parsed.total == 0
    
    parsed = parse_stdout_stderr("   \n\n  ", "")
    assert parsed.total == 0


def test_parse_stdout_stderr_extracts_assertion_diffs():
    """Test that assertion diffs are extracted correctly"""
    stdout = """
tests/test_sample.py::test_assertion FAILED
assert 'foo' == 'bar'
  - bar
  + foo
AssertionError: values don't match
"""
    stderr = ""
    parsed = parse_stdout_stderr(stdout, stderr)
    assert parsed.total == 1
    rec = parsed.failures[0]
    assert rec.assertion_diff is not None
    assert "AssertionError" in rec.assertion_diff


def test_parse_junit_xml_handles_empty(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuite></testsuite>")
    parsed = parse_junit_xml(junit)
    assert parsed.total == 0

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from smart_test_generator.generation.test_generator import IncrementalTestGenerator
from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage, TestableElement
from smart_test_generator.config import Config


class TestIncrementalTestGenerator:
    """Test IncrementalTestGenerator class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.get.return_value = []
        return config

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project root directory."""
        return tmp_path

    @pytest.fixture
    def generator(self, project_root, mock_config):
        """Create an IncrementalTestGenerator instance."""
        with patch('smart_test_generator.generation.test_generator.CodeAnalyzer') as mock_analyzer, \
             patch('smart_test_generator.generation.test_generator.TestMapper') as mock_mapper:
            generator = IncrementalTestGenerator(project_root, mock_config)
            generator.code_analyzer = mock_analyzer.return_value
            generator.test_mapper = mock_mapper.return_value
            return generator

    def test_init_creates_instance_with_required_attributes(self, project_root, mock_config):
        """Test that __init__ creates instance with all required attributes."""
        with patch('smart_test_generator.generation.test_generator.CodeAnalyzer') as mock_analyzer, \
             patch('smart_test_generator.generation.test_generator.TestMapper') as mock_mapper:
            
            generator = IncrementalTestGenerator(project_root, mock_config)
            
            assert generator.project_root == project_root
            assert generator.config == mock_config
            assert generator.code_analyzer is not None
            assert generator.test_mapper is not None
            mock_analyzer.assert_called_once_with(project_root)
            mock_mapper.assert_called_once_with(project_root, mock_config)

    def test_init_with_path_string(self, mock_config):
        """Test that __init__ works with string path."""
        path_str = "/tmp/test"
        with patch('smart_test_generator.generation.test_generator.CodeAnalyzer'), \
             patch('smart_test_generator.generation.test_generator.TestMapper'):
            
            generator = IncrementalTestGenerator(path_str, mock_config)
            
            assert generator.project_root == path_str
            assert generator.config == mock_config

    def test_generate_test_plan_with_no_existing_tests(self, generator):
        """Test generate_test_plan when no existing tests exist."""
        source_file = "src/example.py"
        coverage = None
        
        # Mock dependencies
        generator.test_mapper.find_test_files.return_value = []
        generator.code_analyzer.extract_testable_elements.return_value = [
            TestableElement(name="function1", type="function", filepath="src/example.py", line_number=10, signature="def function1():", complexity=1),
            TestableElement(name="Class1.method1", type="method", filepath="src/example.py", line_number=20, signature="def method1(self):", complexity=2)
        ]
        generator.test_mapper.analyze_test_completeness.return_value = {
            'functions': set(),
            'methods': set()
        }
        
        plan = generator.generate_test_plan(source_file, coverage)
        
        assert isinstance(plan, TestGenerationPlan)
        assert plan.source_file == source_file
        assert plan.existing_test_files == []
        assert len(plan.elements_to_test) == 2
        assert plan.coverage_before is None
        assert plan.estimated_coverage_after == 100.0

    def test_generate_test_plan_with_existing_tests_and_coverage(self, generator):
        """Test generate_test_plan with existing tests and coverage data."""
        source_file = "src/example.py"
        coverage = TestCoverage(
            filepath=source_file,
            line_coverage=60.0,
            branch_coverage=50.0,
            missing_lines=[10, 15, 20],
            covered_functions={"function1"},
            uncovered_functions={"function2"}
        )
        
        generator.test_mapper.find_test_files.return_value = ["tests/test_example.py"]
        generator.code_analyzer.extract_testable_elements.return_value = [
            TestableElement(name="function1", type="function", filepath="src/example.py", line_number=10, signature="def function1():", complexity=1),
            TestableElement(name="function2", type="function", filepath="src/example.py", line_number=15, signature="def function2():", complexity=2)
        ]
        generator.test_mapper.analyze_test_completeness.return_value = {
            'functions': {'function1'},
            'methods': set()
        }
        
        plan = generator.generate_test_plan(source_file, coverage)
        
        assert plan.source_file == source_file
        assert plan.existing_test_files == ["tests/test_example.py"]
        assert len(plan.elements_to_test) == 1
        assert plan.elements_to_test[0].name == "function2"
        assert plan.coverage_before == coverage
        assert plan.estimated_coverage_after > 60.0

    def test_generate_test_plan_with_tested_methods(self, generator):
        """Test generate_test_plan correctly identifies tested methods."""
        source_file = "src/example.py"
        
        generator.test_mapper.find_test_files.return_value = ["tests/test_example.py"]
        generator.code_analyzer.extract_testable_elements.return_value = [
            TestableElement(name="Class1.method1", type="method", filepath="src/example.py", line_number=20, signature="def Class1.method1():", complexity=2),
            TestableElement(name="Class1.method2", type="method", filepath="src/example.py", line_number=30, signature="def Class1.method2():", complexity=1)
        ]
        generator.test_mapper.analyze_test_completeness.return_value = {
            'functions': set(),
            'methods': {'method1'}
        }
        
        plan = generator.generate_test_plan(source_file, None)
        
        assert len(plan.elements_to_test) == 1
        assert plan.elements_to_test[0].name == "Class1.method2"

    def test_generate_test_plan_conservative_mode_without_coverage(self, generator):
        """Test generate_test_plan in conservative mode when no coverage data available."""
        source_file = "src/example.py"
        
        generator.test_mapper.find_test_files.return_value = ["tests/test_example.py"]
        generator.code_analyzer.extract_testable_elements.return_value = [
            TestableElement(name="function1", type="function", filepath="src/example.py", line_number=10, signature="def function1():", complexity=1)
        ]
        generator.test_mapper.analyze_test_completeness.return_value = {
            'functions': set(),
            'methods': set()
        }
        
        with patch.object(generator, '_might_be_tested', return_value=True):
            plan = generator.generate_test_plan(source_file, None)
            
            assert len(plan.elements_to_test) == 0

    def test_estimate_coverage_improvement_no_current_coverage(self, generator):
        """Test _estimate_coverage_improvement with no current coverage."""
        result = generator._estimate_coverage_improvement(10, 5, None)
        assert result == 50.0

    def test_estimate_coverage_improvement_with_current_coverage(self, generator):
        """Test _estimate_coverage_improvement with existing coverage."""
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=60.0,
            branch_coverage=50.0,
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=set()
        )
        
        result = generator._estimate_coverage_improvement(10, 2, coverage)
        # 60 + (2/10 * 100 * 0.8) = 60 + 16 = 76
        assert result == 76.0

    def test_estimate_coverage_improvement_caps_at_100(self, generator):
        """Test _estimate_coverage_improvement caps result at 100%."""
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=95.0,
            branch_coverage=90.0,
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=set()
        )
        
        result = generator._estimate_coverage_improvement(10, 8, coverage)
        assert result == 100.0

    def test_estimate_coverage_improvement_zero_elements(self, generator):
        """Test _estimate_coverage_improvement with zero total elements."""
        result = generator._estimate_coverage_improvement(0, 0, None)
        assert result == 0

    def test_might_be_tested_no_existing_tests(self, generator):
        """Test _might_be_tested returns False when no existing tests."""
        element = TestableElement(name="function1", type="function", filepath="src/example.py", line_number=10, signature="def function1():", complexity=1)
        
        result = generator._might_be_tested(element, [])
        assert result is False

    def test_might_be_tested_finds_test_function_pattern(self, generator):
        """Test _might_be_tested finds test function naming patterns."""
        element = TestableElement(name="calculate_sum", type="function", filepath="src/example.py", line_number=10, signature="def calculate_sum():", complexity=1)
        test_files = ["/tmp/test_example.py"]
        
        test_content = "def test_calculate_sum():\n    pass"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = generator._might_be_tested(element, test_files)
            assert result is True

    def test_might_be_tested_finds_camel_case_pattern(self, generator):
        """Test _might_be_tested finds camelCase test patterns."""
        element = TestableElement(name="calculateSum", type="function", filepath="src/example.py", line_number=10, signature="def calculateSum():", complexity=1)
        test_files = ["/tmp/test_example.py"]
        
        test_content = "def testCalculateSum():\n    pass"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = generator._might_be_tested(element, test_files)
            assert result is True

    def test_might_be_tested_finds_element_name_in_content(self, generator):
        """Test _might_be_tested finds element name mentioned in test content."""
        element = TestableElement(name="special_function", type="function", filepath="src/example.py", line_number=10, signature="def special_function():", complexity=1)
        test_files = ["/tmp/test_example.py"]
        
        test_content = "def test_something():\n    result = special_function()"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = generator._might_be_tested(element, test_files)
            assert result is True

    def test_might_be_tested_handles_method_names(self, generator):
        """Test _might_be_tested correctly handles method names with dots."""
        element = TestableElement(name="Class1.method_name", type="method", filepath="src/example.py", line_number=20, signature="def Class1.method_name():", complexity=2)
        test_files = ["/tmp/test_example.py"]
        
        test_content = "def test_method_name():\n    pass"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = generator._might_be_tested(element, test_files)
            assert result is True

    def test_might_be_tested_handles_file_read_errors(self, generator):
        """Test _might_be_tested handles file read errors gracefully."""
        element = TestableElement(name="function1", type="function", filepath="src/example.py", line_number=10, signature="def function1():", complexity=1)
        test_files = ["/tmp/nonexistent.py"]
        
        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = generator._might_be_tested(element, test_files)
            assert result is False

    def test_might_be_tested_returns_false_when_not_found(self, generator):
        """Test _might_be_tested returns False when element not found in tests."""
        element = TestableElement(name="unfound_function", type="function", filepath="src/example.py", line_number=10, signature="def unfound_function():", complexity=1)
        test_files = ["/tmp/test_example.py"]
        
        test_content = "def test_other_function():\n    pass"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = generator._might_be_tested(element, test_files)
            assert result is False
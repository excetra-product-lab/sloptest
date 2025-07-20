import pytest
from unittest.mock import Mock, patch, mock_open
import os
from smart_test_generator.generation.incremental_generator import IncrementalLLMClient
from smart_test_generator.models.data_models import TestGenerationPlan, TestableElement, WeakSpot
from smart_test_generator.generation.llm_clients import LLMClient
from smart_test_generator.config import Config


class TestIncrementalLLMClient:
    """Test IncrementalLLMClient class."""

    def test_init_stores_base_client_and_config(self):
        """Test that __init__ properly stores base_client and config."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        
        # Act
        client = IncrementalLLMClient(base_client, config)
        
        # Assert
        assert client.base_client is base_client
        assert client.config is config

    def test_init_with_none_base_client(self):
        """Test that __init__ accepts None base_client."""
        # Arrange
        config = Mock(spec=Config)
        
        # Act
        client = IncrementalLLMClient(None, config)
        
        # Assert
        assert client.base_client is None
        assert client.config is config

    def test_init_with_none_config(self):
        """Test that __init__ accepts None config."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        
        # Act
        client = IncrementalLLMClient(base_client, None)
        
        # Assert
        assert client.base_client is base_client
        assert client.config is None

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    def test_generate_contextual_tests_with_empty_plan_list(self, mock_get_system_prompt):
        """Test that generate_contextual_tests returns empty dict for empty plan list."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        # Act
        result = client.generate_contextual_tests([], "directory_structure")
        
        # Assert
        assert result == {}
        mock_get_system_prompt.assert_not_called()
        base_client.generate_unit_tests.assert_not_called()

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    def test_generate_contextual_tests_skips_plans_without_elements(self, mock_get_system_prompt):
        """Test that generate_contextual_tests skips plans with no elements_to_test."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = []
        
        # Act
        result = client.generate_contextual_tests([plan], "directory_structure")
        
        # Assert
        assert result == {}
        mock_get_system_prompt.assert_not_called()
        base_client.generate_unit_tests.assert_not_called()

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    @patch('builtins.open', new_callable=mock_open, read_data='source code content')
    def test_generate_contextual_tests_with_valid_plan(self, mock_file, mock_get_system_prompt):
        """Test that generate_contextual_tests processes valid plan correctly."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        element = Mock(spec=TestableElement)
        element.type = "method"
        element.name = "test_method"
        element.line_number = 10
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = [element]
        plan.existing_test_files = []
        plan.source_file = "/path/to/source.py"
        
        mock_get_system_prompt.return_value = "system prompt"
        base_client.generate_unit_tests.return_value = {"/path/to/test.py": "test content"}
        
        # Mock the private methods
        client._read_existing_tests = Mock(return_value="")
        client._create_contextual_prompt = Mock(return_value="contextual prompt")
        client._create_focused_xml = Mock(return_value="<xml>content</xml>")
        
        # Act
        result = client.generate_contextual_tests([plan], "directory_structure")
        
        # Assert
        assert result == {"/path/to/source.py": "test content"}
        client._read_existing_tests.assert_called_once_with([])
        client._create_contextual_prompt.assert_called_once_with(plan, "")
        client._create_focused_xml.assert_called_once_with(plan)
        base_client.generate_unit_tests.assert_called_once_with(
            "contextual prompt", "<xml>content</xml>", "directory_structure", None, None
        )

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    @patch('builtins.open', new_callable=mock_open, read_data='source code content')
    def test_generate_contextual_tests_with_existing_test_files(self, mock_file, mock_get_system_prompt):
        """Test that generate_contextual_tests merges with existing test files."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        element = Mock(spec=TestableElement)
        element.type = "method"
        element.name = "test_method"
        element.line_number = 10
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = [element]
        plan.existing_test_files = ["/path/to/existing_test.py"]
        plan.source_file = "/path/to/source.py"
        
        mock_get_system_prompt.return_value = "system prompt"
        base_client.generate_unit_tests.return_value = {"/path/to/test.py": "new test content"}
        
        # Mock the private methods
        client._read_existing_tests = Mock(return_value="existing content")
        client._create_contextual_prompt = Mock(return_value="contextual prompt")
        client._create_focused_xml = Mock(return_value="<xml>content</xml>")
        client._merge_tests = Mock(return_value="merged content")
        
        # Act
        result = client.generate_contextual_tests([plan], "directory_structure")
        
        # Assert
        assert result == {"/path/to/source.py": "merged content"}
        client._merge_tests.assert_called_once_with("/path/to/existing_test.py", "new test content")

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    def test_generate_single_file_test_with_no_elements(self, mock_get_system_prompt):
        """Test that generate_single_file_test returns None when no elements to test."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = []
        
        # Act
        result = client.generate_single_file_test(plan, "directory_structure")
        
        # Assert
        assert result is None
        mock_get_system_prompt.assert_not_called()
        base_client.generate_unit_tests.assert_not_called()

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    @patch('builtins.open', new_callable=mock_open, read_data='source code content')
    def test_generate_single_file_test_with_valid_plan(self, mock_file, mock_get_system_prompt):
        """Test that generate_single_file_test generates test for valid plan."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        element = Mock(spec=TestableElement)
        element.type = "method"
        element.name = "test_method"
        element.line_number = 10
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = [element]
        plan.existing_test_files = []
        plan.source_file = "/path/to/source.py"
        
        mock_get_system_prompt.return_value = "system prompt"
        base_client.generate_unit_tests.return_value = {"/path/to/test.py": "test content"}
        
        # Mock the private methods
        client._read_existing_tests = Mock(return_value="")
        client._create_contextual_prompt = Mock(return_value="contextual prompt")
        client._create_focused_xml = Mock(return_value="<xml>content</xml>")
        
        # Act
        result = client.generate_single_file_test(plan, "directory_structure")
        
        # Assert
        assert result == "test content"
        base_client.generate_unit_tests.assert_called_once_with(
            "contextual prompt", "<xml>content</xml>", "directory_structure", ["/path/to/source.py"], None
        )

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    @patch('builtins.open', new_callable=mock_open, read_data='source code content')
    def test_generate_single_file_test_with_existing_test_files(self, mock_file, mock_get_system_prompt):
        """Test that generate_single_file_test merges with existing test files."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        element = Mock(spec=TestableElement)
        element.type = "method"
        element.name = "test_method"
        element.line_number = 10
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = [element]
        plan.existing_test_files = ["/path/to/existing_test.py"]
        plan.source_file = "/path/to/source.py"
        
        mock_get_system_prompt.return_value = "system prompt"
        base_client.generate_unit_tests.return_value = {"/path/to/test.py": "new test content"}
        
        # Mock the private methods
        client._read_existing_tests = Mock(return_value="existing content")
        client._create_contextual_prompt = Mock(return_value="contextual prompt")
        client._create_focused_xml = Mock(return_value="<xml>content</xml>")
        client._merge_tests = Mock(return_value="merged content")
        
        # Act
        result = client.generate_single_file_test(plan, "directory_structure")
        
        # Assert
        assert result == "merged content"
        client._merge_tests.assert_called_once_with("/path/to/existing_test.py", "new test content")

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    @patch('builtins.open', new_callable=mock_open, read_data='source code content')
    def test_generate_single_file_test_with_custom_source_files(self, mock_file, mock_get_system_prompt):
        """Test that generate_single_file_test uses custom source_files parameter."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        element = Mock(spec=TestableElement)
        element.type = "method"
        element.name = "test_method"
        element.line_number = 10
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = [element]
        plan.existing_test_files = []
        plan.source_file = "/path/to/source.py"
        
        mock_get_system_prompt.return_value = "system prompt"
        base_client.generate_unit_tests.return_value = {"/path/to/test.py": "test content"}
        
        # Mock the private methods
        client._read_existing_tests = Mock(return_value="")
        client._create_contextual_prompt = Mock(return_value="contextual prompt")
        client._create_focused_xml = Mock(return_value="<xml>content</xml>")
        
        custom_source_files = ["/custom/source1.py", "/custom/source2.py"]
        
        # Act
        result = client.generate_single_file_test(plan, "directory_structure", custom_source_files, "/project/root")
        
        # Assert
        assert result == "test content"
        base_client.generate_unit_tests.assert_called_once_with(
            "contextual prompt", "<xml>content</xml>", "directory_structure", custom_source_files, "/project/root"
        )

    @patch('smart_test_generator.generation.incremental_generator.get_system_prompt')
    @patch('builtins.open', new_callable=mock_open, read_data='source code content')
    def test_generate_single_file_test_returns_none_when_no_tests_generated(self, mock_file, mock_get_system_prompt):
        """Test that generate_single_file_test returns None when no tests are generated."""
        # Arrange
        base_client = Mock(spec=LLMClient)
        config = Mock(spec=Config)
        client = IncrementalLLMClient(base_client, config)
        
        element = Mock(spec=TestableElement)
        element.type = "method"
        element.name = "test_method"
        element.line_number = 10
        
        plan = Mock(spec=TestGenerationPlan)
        plan.elements_to_test = [element]
        plan.existing_test_files = []
        plan.source_file = "/path/to/source.py"
        
        mock_get_system_prompt.return_value = "system prompt"
        base_client.generate_unit_tests.return_value = {}  # No tests generated
        
        # Mock the private methods
        client._read_existing_tests = Mock(return_value="")
        client._create_contextual_prompt = Mock(return_value="contextual prompt")
        client._create_focused_xml = Mock(return_value="<xml>content</xml>")
        
        # Act
        result = client.generate_single_file_test(plan, "directory_structure")
        
        # Assert
        assert result is None
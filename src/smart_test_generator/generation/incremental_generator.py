"""Incremental LLM-based test generation."""

import os
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional

from smart_test_generator.models.data_models import TestGenerationPlan
from smart_test_generator.generation.llm_clients import LLMClient, get_system_prompt
from smart_test_generator.config import Config
from smart_test_generator.utils.prompt_loader import get_prompt_loader

logger = logging.getLogger(__name__)


class IncrementalLLMClient:
    """Enhanced LLM client that generates tests based on gaps."""

    def __init__(self, base_client: LLMClient, config: Config):
        self.base_client = base_client
        self.config = config

    def generate_contextual_tests(self, test_plans: List[TestGenerationPlan],
                                  directory_structure: str, source_files: List[str] = None, 
                                  project_root: str = None) -> Dict[str, str]:
        """Generate tests with awareness of existing tests."""
        all_tests = {}

        for plan in test_plans:
            if not plan.elements_to_test:
                continue

            # Read existing test files for context
            existing_test_content = self._read_existing_tests(plan.existing_test_files)

            # Generate enhanced prompt
            system_prompt = self._create_contextual_prompt(plan, existing_test_content)

            # Create focused XML for just the elements needing tests
            xml_content = self._create_focused_xml(plan)

            # Generate tests
            tests = self.base_client.generate_unit_tests(system_prompt, xml_content, directory_structure, 
                                                        source_files, project_root)

            # Map test content back to source file path (not test file path)
            # The test generation service expects source file paths as keys
            test_content = None
            for filepath, content in tests.items():
                if plan.existing_test_files:
                    # Merge with existing test file
                    merged_content = self._merge_tests(plan.existing_test_files[0], content)
                    test_content = merged_content
                else:
                    test_content = content
                # Use the first (and usually only) test content
                break
            
            # Store using source file path as key (not test file path)
            if test_content:
                all_tests[plan.source_file] = test_content

        return all_tests

    def generate_single_file_test(self, plan: TestGenerationPlan, 
                                  directory_structure: str, source_files: List[str] = None, 
                                  project_root: str = None) -> Optional[str]:
        """Generate tests for a single file immediately."""
        if not plan.elements_to_test:
            return None

        # Read existing test files for context
        existing_test_content = self._read_existing_tests(plan.existing_test_files)

        # Generate enhanced prompt
        system_prompt = self._create_contextual_prompt(plan, existing_test_content)

        # Create focused XML for just the elements needing tests
        xml_content = self._create_focused_xml(plan)

        # Generate tests for single file
        tests = self.base_client.generate_unit_tests(system_prompt, xml_content, directory_structure, 
                                                    source_files or [plan.source_file], project_root)

        # Return the first test content found
        for filepath, content in tests.items():
            if plan.existing_test_files:
                # Merge with existing test file
                return self._merge_tests(plan.existing_test_files[0], content)
            else:
                return content
        
        return None

    def refine_tests(self, request: Dict) -> str:
        """Delegate test refinement to the underlying base client."""
        return self.base_client.refine_tests(request)

    def _read_existing_tests(self, test_files: List[str]) -> str:
        """Read existing test files for context."""
        content = []
        for test_file in test_files[:2]:  # Limit to first 2 files for context
            try:
                with open(test_file, 'r') as f:
                    content.append(f"=== {os.path.basename(test_file)} ===\n{f.read()}\n")
            except:
                pass
        return "\n".join(content)

    def _create_contextual_prompt(self, plan: TestGenerationPlan, existing_tests: str) -> str:
        """Create a prompt that includes context about existing tests and quality insights."""
        base_prompt = get_system_prompt(self.config)
        prompt_loader = get_prompt_loader()
        
        # Focus on specific untested elements only
        untested_elements = [f"{e.type} {e.name}" for e in plan.elements_to_test]
        
        # Check if we should use 2025 guidelines for context formatting
        use_2025_guidelines = self.config.get('prompt_engineering.use_2025_guidelines', True)
        
        # Build quality target if available
        quality_target = getattr(plan, 'quality_score_target', None)
        
        # Build mutation guidance data if available
        mutation_guidance = None
        if hasattr(plan, 'weak_mutation_spots') or hasattr(plan, 'mutation_score_target'):
            mutation_guidance = {
                'weak_mutation_spots': getattr(plan, 'weak_mutation_spots', []),
                'mutation_score_target': getattr(plan, 'mutation_score_target', None)
            }
        
        # Get contextual prompt from loader
        return prompt_loader.get_contextual_prompt(
            base_prompt=base_prompt,
            untested_elements=untested_elements,
            existing_tests=existing_tests,
            use_2025_format=use_2025_guidelines,
            quality_target=quality_target,
            mutation_guidance=mutation_guidance,
            config=self.config
        )

    # Quality and mutation guidance methods removed - now handled by PromptLoader

    def _create_focused_xml(self, plan: TestGenerationPlan) -> str:
        """Create XML content focused only on elements needing tests."""
        # Read the source file
        with open(plan.source_file, 'r') as f:
            content = f.read()

        # Create a simplified XML with just the necessary content
        root = ET.Element("codebase")
        file_elem = ET.SubElement(root, "file")
        file_elem.set("filename", os.path.basename(plan.source_file))
        file_elem.set("filepath", os.path.relpath(plan.source_file))

        # Add markers for elements to test
        markers = []
        for element in plan.elements_to_test:
            markers.append(f"# TODO: Generate test for {element.type} {element.name} at line {element.line_number}")

        file_elem.text = "\n".join(markers) + "\n\n" + content

        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")

    def _merge_tests(self, existing_test_file: str, new_tests: str) -> str:
        """Merge new tests with existing test file."""
        try:
            with open(existing_test_file, 'r') as f:
                existing_content = f.read()

            # Simple merge: append new tests to the end
            # In a real implementation, this would be more sophisticated
            merged = existing_content.rstrip() + "\n\n" + new_tests
            return merged

        except:
            return new_tests

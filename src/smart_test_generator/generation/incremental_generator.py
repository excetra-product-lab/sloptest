"""Incremental LLM-based test generation."""

import os
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional

from smart_test_generator.models.data_models import TestGenerationPlan
from smart_test_generator.generation.llm_clients import LLMClient, get_system_prompt
from smart_test_generator.config import Config

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

        # Focus on specific untested elements only
        untested_elements = [f"{e.type} {e.name}" for e in plan.elements_to_test]
        
        # Check if we should use 2025 guidelines for context formatting
        use_2025_guidelines = self.config.get('prompt_engineering.use_2025_guidelines', True)
        
        if use_2025_guidelines:
            context = f"""\n\nCONTEXT - INCREMENTAL TESTING:
Generate tests ONLY for these untested elements: {', '.join(untested_elements)}

Existing test style (match this):
{existing_tests[:800]}...

Do not regenerate existing tests. Focus exclusively on the listed elements."""
        else:
            # Legacy format with more verbose XML-style structure
            context = f"""\n\nIMPORTANT CONTEXT:
This file already has some tests. You should ONLY generate tests for the following untested elements:

{', '.join(untested_elements)}

The existing test file(s) use the following patterns and style:
{existing_tests[:1000]}...

Please match the existing testing style and patterns. Only generate tests for the elements listed above.
Do not regenerate tests for elements that are already tested."""

        # Add targeted quality guidance (more concise)
        quality_guidance = self._generate_quality_guidance(plan)
        if quality_guidance:
            context += f"\n\n{quality_guidance}"

        # Add focused mutation insights
        mutation_guidance = self._generate_mutation_guidance(plan)
        if mutation_guidance:
            context += f"\n\n{mutation_guidance}"

        return base_prompt + context

    def _generate_quality_guidance(self, plan: TestGenerationPlan) -> str:
        """Generate concise quality-focused guidance."""
        if not hasattr(plan, 'quality_score_target') or not plan.quality_score_target:
            return ""
        
        target = plan.quality_score_target
        
        if target >= 90:
            return f"""QUALITY TARGET: {target:.0f}% (HIGH STANDARD)
- Use exact assertions with specific expected values
- Test all edge cases: None, empty collections, boundaries
- Include comprehensive error condition testing
- Ensure test independence with proper fixtures"""
        
        elif target >= 75:
            return f"""QUALITY TARGET: {target:.0f}% (GOOD STANDARD)
- Test success and error conditions
- Include edge cases and boundary values
- Use specific assertions over generic ones"""
        
        else:
            return f"""QUALITY TARGET: {target:.0f}% (BASIC STANDARD)
- Focus on common use cases and clear assertions
- Include basic error condition testing"""

    def _generate_mutation_guidance(self, plan: TestGenerationPlan) -> str:
        """Generate focused mutation testing guidance."""
        if not hasattr(plan, 'weak_mutation_spots') or not plan.weak_mutation_spots:
            return ""
        
        guidance = ["MUTATION TESTING - Address these weak patterns:"]
        
        # Group and limit to most important patterns
        weak_spots_by_type = {}
        for weak_spot in plan.weak_mutation_spots[:3]:  # Limit to top 3
            mut_type = weak_spot.mutation_type.value
            if mut_type not in weak_spots_by_type:
                weak_spots_by_type[mut_type] = []
            weak_spots_by_type[mut_type].append(weak_spot)
        
        for mut_type, spots in weak_spots_by_type.items():
            type_name = mut_type.replace('_', ' ').title()
            
            if mut_type == "arithmetic_operator":
                guidance.append(f"• {type_name}: Test exact arithmetic results, verify different operators fail")
            elif mut_type == "comparison_operator":
                guidance.append(f"• {type_name}: Test boundary conditions (equal, just above/below)")
            elif mut_type == "logical_operator":
                guidance.append(f"• {type_name}: Test both AND/OR branches with scenarios where logic matters")
            elif mut_type == "constant_value":
                guidance.append(f"• {type_name}: Use specific expected values, test boundary constants")
            elif mut_type == "boundary_value":
                guidance.append(f"• {type_name}: Add comprehensive off-by-one tests")
            
            # Add top suggestion if available
            if spots and spots[0].suggested_tests:
                guidance.append(f"  Suggestion: {spots[0].suggested_tests[0]}")
        
        if hasattr(plan, 'mutation_score_target') and plan.mutation_score_target:
            guidance.append(f"\nTARGET MUTATION SCORE: {plan.mutation_score_target:.0f}%")
        
        return "\n".join(guidance)

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

"""Incremental LLM-based test generation."""

import os
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
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
        """Create XML content with full codebase context for better test generation."""
        root = ET.Element("codebase")
        
        # Include the primary file being tested (with markers)
        primary_file = ET.SubElement(root, "file")
        primary_file.set("filename", os.path.basename(plan.source_file))
        primary_file.set("filepath", os.path.relpath(plan.source_file))
        primary_file.set("primary", "true")  # Mark as the primary file being tested
        
        # Read the primary source file
        with open(plan.source_file, 'r') as f:
            content = f.read()
        
        # Add markers for elements to test
        markers = []
        for element in plan.elements_to_test:
            markers.append(f"# TODO: Generate test for {element.type} {element.name} at line {element.line_number}")
        
        primary_file.text = "\n".join(markers) + "\n\n" + content
        
        # Include full codebase context if enabled
        include_full_context = self.config.get('test_generation.include_full_codebase_context', True)
        if include_full_context:
            context_files = self._get_codebase_context_files(plan.source_file)
            max_context_files = self.config.get('test_generation.max_context_files', 50)
            
            # Limit the number of context files to avoid token overload
            if len(context_files) > max_context_files:
                logger.info(f"Limiting codebase context to {max_context_files} files (found {len(context_files)})")
                context_files = context_files[:max_context_files]
            
            for context_file_path in context_files:
                try:
                    context_file = ET.SubElement(root, "file")
                    context_file.set("filename", os.path.basename(context_file_path))
                    context_file.set("filepath", os.path.relpath(context_file_path))
                    context_file.set("context", "true")  # Mark as context file
                    
                    with open(context_file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        # Skip very large files to avoid token overload
                        max_file_size = self.config.get('test_generation.max_context_file_size', 10000)
                        if len(file_content) > max_file_size:
                            context_file.text = f"# File too large ({len(file_content):,} chars) - truncated\n{file_content[:max_file_size//2]}...\n{file_content[-max_file_size//2:]}"
                        else:
                            context_file.text = file_content
                except Exception as e:
                    logger.warning(f"Failed to read context file {context_file_path}: {e}")
                    # Include a placeholder for failed files
                    context_file = ET.SubElement(root, "file")
                    context_file.set("filename", os.path.basename(context_file_path))
                    context_file.set("filepath", os.path.relpath(context_file_path))
                    context_file.set("context", "true")
                    context_file.text = f"# Unable to read file: {e}"

        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")

    def _get_codebase_context_files(self, primary_source_file: str) -> List[str]:
        """Get list of source files to include as codebase context."""
        from smart_test_generator.utils.parser import PythonCodebaseParser
        
        # Get project root from primary file path
        primary_path = Path(primary_source_file)
        
        # Try to determine project root - look for common markers
        project_root = primary_path.parent
        while project_root.parent != project_root:
            if any((project_root / marker).exists() for marker in ['pyproject.toml', 'setup.py', '.git', 'requirements.txt']):
                break
            project_root = project_root.parent
        
        # Use the parser to find all Python files
        parser = PythonCodebaseParser(str(project_root), self.config)
        all_files = parser.find_python_files()
        
        # Filter and prioritize files
        context_files = []
        primary_dir = Path(primary_source_file).parent
        
        # Priority 1: Files in the same directory as the primary file
        same_dir_files = [f for f in all_files if Path(f).parent == primary_dir and f != primary_source_file]
        context_files.extend(same_dir_files)
        
        # Priority 2: Files that are imported by the primary file (analyze imports)
        try:
            imported_files = self._analyze_imports(primary_source_file, project_root, all_files)
            for imp_file in imported_files:
                if imp_file not in context_files and imp_file != primary_source_file:
                    context_files.append(imp_file)
        except Exception as e:
            logger.debug(f"Failed to analyze imports for {primary_source_file}: {e}")
        
        # Priority 3: Other files in the same package
        primary_package = self._get_package_name(primary_source_file, project_root)
        if primary_package:
            package_files = [f for f in all_files 
                           if self._get_package_name(f, project_root) == primary_package 
                           and f not in context_files 
                           and f != primary_source_file]
            context_files.extend(package_files)
        
        # Priority 4: Remaining files (up to limit)
        remaining_files = [f for f in all_files if f not in context_files and f != primary_source_file]
        context_files.extend(remaining_files)
        
        logger.debug(f"Selected {len(context_files)} context files for {primary_source_file}")
        return context_files

    def _analyze_imports(self, source_file: str, project_root: Path, all_files: List[str]) -> List[str]:
        """Analyze imports in source file and find corresponding files."""
        import ast
        imported_files = []
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # Convert module path to file path
                    if node.module.startswith('.'):
                        # Relative import - resolve relative to current file
                        module_file = self._resolve_relative_import(source_file, node.module, project_root)
                    else:
                        # Absolute import
                        module_file = self._resolve_absolute_import(node.module, project_root, all_files)
                    
                    if module_file and module_file in all_files:
                        imported_files.append(module_file)
                        
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_file = self._resolve_absolute_import(alias.name, project_root, all_files)
                        if module_file and module_file in all_files:
                            imported_files.append(module_file)
        
        except Exception as e:
            logger.debug(f"Error analyzing imports in {source_file}: {e}")
        
        return imported_files

    def _resolve_relative_import(self, source_file: str, module: str, project_root: Path) -> str:
        """Resolve relative import to file path."""
        source_path = Path(source_file)
        source_dir = source_path.parent
        
        # Handle relative imports like .module or ..module
        parts = module.split('.')
        levels_up = len([p for p in parts if p == ''])
        module_parts = [p for p in parts if p]
        
        # Go up the specified number of levels
        target_dir = source_dir
        for _ in range(levels_up - 1):  # -1 because one dot means current package
            target_dir = target_dir.parent
        
        # Build the module path
        if module_parts:
            for part in module_parts:
                target_dir = target_dir / part
            target_file = target_dir / '__init__.py'
            if not target_file.exists():
                target_file = target_dir.with_suffix('.py')
                if target_file.exists():
                    return str(target_file)
        
        return ""

    def _resolve_absolute_import(self, module: str, project_root: Path, all_files: List[str]) -> str:
        """Resolve absolute import to file path within the project."""
        module_parts = module.split('.')
        
        # Try different combinations to find the file
        for i in range(len(module_parts), 0, -1):
            potential_path = project_root
            for part in module_parts[:i]:
                potential_path = potential_path / part
            
            # Try as a package (__init__.py)
            init_file = potential_path / '__init__.py'
            if str(init_file) in all_files:
                return str(init_file)
            
            # Try as a module (.py file)
            module_file = potential_path.with_suffix('.py')
            if str(module_file) in all_files:
                return str(module_file)
        
        return ""

    def _get_package_name(self, file_path: str, project_root: Path) -> str:
        """Get package name for a file."""
        try:
            rel_path = Path(file_path).relative_to(project_root)
            return '.'.join(rel_path.parts[:-1])  # Exclude filename
        except:
            return ""

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

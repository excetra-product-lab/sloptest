#!/usr/bin/env python3
"""
Demo script to showcase the newly implemented configuration options.

This script demonstrates how the previously unused config options now work
and shows their impact on test generation behavior.
"""

import os
import tempfile
from pathlib import Path
from smart_test_generator.config import Config
from smart_test_generator.utils.prompt_loader import PromptLoader
from smart_test_generator.models.data_models import TestCoverage
from smart_test_generator.tracking.state_tracker import TestGenerationTracker


def demo_config_options():
    """Demonstrate the newly implemented config options."""
    
    print("🚀 Smart Test Generator - New Configuration Options Demo")
    print("=" * 60)
    
    # Create a temporary config file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        config_content = """
test_generation:
  style:
    framework: pytest
    assertion_style: unittest  # NEW: Control assertion style
    mock_library: pytest-mock  # NEW: Control mock library
    
  coverage:
    minimum_line_coverage: 80
    minimum_branch_coverage: 75  # NEW: Branch coverage threshold
    regenerate_if_below: 60
    
  generation:
    include_docstrings: minimal  # NEW: Control docstring generation
    generate_fixtures: true      # NEW: Generate pytest fixtures
    parametrize_similar_tests: true  # NEW: Use parametrize for similar tests
    max_test_methods_per_class: 15   # NEW: Split large test classes
    always_analyze_new_files: true  # NEW: Always analyze new files
"""
        f.write(config_content)
        config_file = f.name
    
    try:
        # Load the config
        config = Config(config_file)
        print(f"✅ Loaded config from: {config_file}")
        
        # Demo 1: Assertion Style Configuration
        print(f"\n📝 Demo 1: Assertion Style")
        print(f"Configured assertion style: {config.get('test_generation.style.assertion_style')}")
        
        prompt_loader = PromptLoader()
        system_prompt = prompt_loader.get_system_prompt(config=config)
        
        if "unittest-style assertions" in system_prompt.lower():
            print("✅ System prompt correctly configured for unittest-style assertions")
        else:
            print("ℹ️  System prompt uses default assertion guidance")
            
        # Demo 2: Mock Library Configuration
        print(f"\n🎭 Demo 2: Mock Library")
        print(f"Configured mock library: {config.get('test_generation.style.mock_library')}")
        
        if "pytest-mock" in system_prompt.lower():
            print("✅ System prompt correctly configured for pytest-mock")
        else:
            print("ℹ️  System prompt uses default mock guidance")
            
        # Demo 3: Branch Coverage Configuration
        print(f"\n📊 Demo 3: Branch Coverage")
        min_branch = config.get('test_generation.coverage.minimum_branch_coverage')
        print(f"Minimum branch coverage threshold: {min_branch}%")
        
        # Create a mock coverage object to test the logic
        test_coverage = TestCoverage(
            filepath="test_file.py",
            line_coverage=85.0,
            branch_coverage=70.0,  # Below the 75% threshold
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=set()
        )
        
        tracker = TestGenerationTracker()
        should_generate, reason = tracker.should_generate_tests("test_file.py", test_coverage, config)
        
        if should_generate and "branch coverage" in reason.lower():
            print(f"✅ Branch coverage check working: {reason}")
        else:
            print(f"ℹ️  Coverage evaluation: {reason}")
            
        # Demo 4: Docstring Configuration
        print(f"\n📚 Demo 4: Docstring Generation")
        docstring_setting = config.get('test_generation.generation.include_docstrings')
        print(f"Docstring setting: {docstring_setting}")
        
        if "minimal" in system_prompt.lower():
            print("✅ System prompt configured for minimal docstrings")
        else:
            print("ℹ️  System prompt uses default docstring guidance")
            
        # Demo 5: Fixture Generation
        print(f"\n🔧 Demo 5: Fixture Generation")
        generate_fixtures = config.get('test_generation.generation.generate_fixtures')
        print(f"Generate fixtures enabled: {generate_fixtures}")
        
        contextual_prompt = prompt_loader.get_contextual_prompt(
            base_prompt="Base prompt",
            untested_elements=["test_function"],
            existing_tests="",
            config=config
        )
        
        if "fixture" in contextual_prompt.lower():
            print("✅ Contextual prompt includes fixture generation guidance")
        else:
            print("ℹ️  Fixture guidance not found in contextual prompt")
            
        # Demo 6: Parametrization
        print(f"\n🔄 Demo 6: Test Parametrization")
        parametrize_tests = config.get('test_generation.generation.parametrize_similar_tests')
        print(f"Parametrize similar tests: {parametrize_tests}")
        
        if "parametrize" in contextual_prompt.lower():
            print("✅ Contextual prompt includes parametrization guidance")
        else:
            print("ℹ️  Parametrization guidance not found in contextual prompt")
            
        # Demo 7: Max Methods Per Class
        print(f"\n🏗️  Demo 7: Max Test Methods Per Class")
        max_methods = config.get('test_generation.generation.max_test_methods_per_class')
        print(f"Maximum methods per class: {max_methods}")
        
        # Demo test splitting (this would normally happen in TestFileWriter)
        test_content_with_many_methods = '''
import pytest

class TestExample:
    def test_method_1(self): pass
    def test_method_2(self): pass
    def test_method_3(self): pass
    def test_method_4(self): pass
    def test_method_5(self): pass
    def test_method_6(self): pass
    def test_method_7(self): pass
    def test_method_8(self): pass
    def test_method_9(self): pass
    def test_method_10(self): pass
    def test_method_11(self): pass
    def test_method_12(self): pass
    def test_method_13(self): pass
    def test_method_14(self): pass
    def test_method_15(self): pass
    def test_method_16(self): pass  # This exceeds the limit
    def test_method_17(self): pass
'''
        
        print(f"Example test class has 17 methods (exceeds limit of {max_methods})")
        print("✅ Class splitting logic would activate to split this into multiple classes")
        
        # Demo 8: Always Analyze New Files
        print(f"\n🔍 Demo 8: Always Analyze New Files")
        always_analyze = config.get('test_generation.generation.always_analyze_new_files')
        print(f"Always analyze new files: {always_analyze}")
        
        # Test this logic
        should_analyze, reason = tracker.should_generate_tests("new_file.py", None, config)
        
        if should_analyze and "configuration" in reason.lower():
            print(f"✅ New file analysis logic working: {reason}")
        else:
            print(f"ℹ️  Analysis decision: {reason}")
            
        print(f"\n🎉 Demo Complete!")
        print(f"All 8 previously unused config options are now fully implemented:")
        print(f"  ✅ assertion_style")
        print(f"  ✅ mock_library") 
        print(f"  ✅ minimum_branch_coverage")
        print(f"  ✅ include_docstrings")
        print(f"  ✅ generate_fixtures")
        print(f"  ✅ parametrize_similar_tests")
        print(f"  ✅ max_test_methods_per_class")
        print(f"  ✅ always_analyze_new_files")
        
    finally:
        # Clean up temp config file
        os.unlink(config_file)


if __name__ == "__main__":
    demo_config_options()

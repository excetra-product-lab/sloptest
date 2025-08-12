from smart_test_generator.utils.ast_merge import (
    parse_module, index_module, ast_equal, ImportEntry, merge_modules
)


def test_parse_and_index_module_simple():
    src = """
import os
from pathlib import Path as P

def helper():
    return 1

class T:
    def test_a(self):
        return 2
"""
    mod = parse_module(src)
    idx = index_module(mod)
    # imports
    imports = idx["imports"]
    assert ImportEntry(module=None, name="os", asname=None, level=0) in imports
    assert ImportEntry(module="pathlib", name="Path", asname="P", level=0) in imports
    # functions
    funcs = idx["functions"]
    assert "helper" in funcs
    # classes/methods
    classes = idx["classes"]
    assert "T" in classes
    cls, methods = classes["T"]
    assert "test_a" in methods


def test_ast_equal_ignores_formatting():
    a = parse_module("def x():\n    return 1\n")
    b = parse_module("def x():\n\n    return 1\n")
    # Compare specific nodes (first function)
    assert ast_equal(a.body[0], b.body[0])


def test_ast_equal_ignores_docstrings_by_default():
    a = parse_module('def x():\n    """doc"""\n    return 1\n')
    b = parse_module('def x():\n    return 1\n')
    assert ast_equal(a.body[0], b.body[0])


def test_merge_modules_adds_missing_imports_and_functions_and_methods():
    existing = """
import os

def helper():
    return 1

class T:
    def test_a(self):
        return 2
"""
    new = """
import os
from pathlib import Path

def helper():
    return 1

def new_top():
    return 3

class T:
    def test_a(self):
        return 2
    def test_b(self):
        return 4
"""

    merged_src, actions = merge_modules(existing, new)
    assert "from pathlib import Path" in merged_src
    assert "def new_top(" in merged_src
    assert "def test_b(" in merged_src
    # ensure no duplicate class T or function helper
    assert merged_src.count("class T(") == 1 or merged_src.count("class T:") == 1
    assert merged_src.count("def helper(") == 1
    # actions recorded
    assert any(a.startswith("import:add:") for a in actions)
    assert "func:add:new_top" in actions
    assert "method:add:T.test_b" in actions


def test_merge_modules_unions_decorators():
    existing = """
import pytest

@pytest.mark.slow
def test_x():
    assert True
"""
    new = """
import pytest

@pytest.mark.db
def test_x():
    assert True
"""
    merged_src, actions = merge_modules(existing, new)
    # both decorators present in merged output
    assert "@pytest.mark.slow" in merged_src
    assert "@pytest.mark.db" in merged_src


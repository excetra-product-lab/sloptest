"""Utilities for parsing Python modules and performing structure-aware comparisons.

This module provides:
- parse_module(source_text) -> (ast.Module, source_text)
- index_module(module_ast) -> dict of imports/classes/functions
- ast_equal(a, b) -> structural equality ignoring formatting/positions

These utilities are the foundation for an AST-based merge strategy for tests.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import copy
import ast
import difflib


@dataclass(frozen=True)
class ImportEntry:
    module: Optional[str]  # None for plain "import x"
    name: str
    asname: Optional[str]
    level: int = 0  # For from-imports; 0 for absolute


def parse_module(source_text: str) -> ast.Module:
    """Parse Python source text into an AST module.

    Raises SyntaxError if the source is invalid. Caller should handle fallback.
    """
    return ast.parse(source_text)


def _normalize_imports(module: ast.Module) -> Set[ImportEntry]:
    entries: Set[ImportEntry] = set()
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                entries.add(ImportEntry(module=None, name=alias.name, asname=alias.asname, level=0))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            level = int(node.level or 0)
            for alias in node.names:
                entries.add(ImportEntry(module=mod, name=alias.name, asname=alias.asname, level=level))
    return entries


def _collect_module_level_functions(module: ast.Module) -> Dict[str, ast.AST]:
    funcs: Dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    return funcs


def _collect_classes(module: ast.Module) -> Dict[str, Tuple[ast.ClassDef, Dict[str, ast.AST]]]:
    classes: Dict[str, Tuple[ast.ClassDef, Dict[str, ast.AST]]] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            methods: Dict[str, ast.AST] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[item.name] = item
            classes[node.name] = (node, methods)
    return classes


def index_module(module: ast.Module) -> Dict[str, object]:
    """Create an index of imports, classes(with methods), and module-level functions.

    Returns a dictionary with keys: imports, classes, functions.
    """
    return {
        "imports": _normalize_imports(module),
        "classes": _collect_classes(module),
        "functions": _collect_module_level_functions(module),
    }


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """Return a shallow copy of node with docstrings removed from functions/classes."""
    node = copy.deepcopy(node)
    def strip_in_body(body: List[ast.stmt]) -> List[ast.stmt]:
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
            return body[1:]
        return body
    # Function or AsyncFunction
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.body = strip_in_body(node.body)
        return node
    # Class: strip class docstring and function docstrings within
    if isinstance(node, ast.ClassDef):
        node.body = strip_in_body(node.body)
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.body[i] = _strip_docstrings(stmt)  # type: ignore
        return node
    return node


def ast_equal(a: ast.AST, b: ast.AST, *, ignore_docstrings: bool = True) -> bool:
    """Compare two AST nodes structurally, ignoring non-semantic attributes.

    Uses ast.dump with include_attributes=False for a normalized representation.
    """
    try:
        if ignore_docstrings:
            a = _strip_docstrings(a)
            b = _strip_docstrings(b)
        return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    except Exception:
        return False


def _import_entry_to_node(entry: ImportEntry) -> ast.stmt:
    if entry.module is None:
        return ast.Import(names=[ast.alias(name=entry.name, asname=entry.asname)])
    return ast.ImportFrom(
        module=entry.module,
        names=[ast.alias(name=entry.name, asname=entry.asname)],
        level=entry.level,
    )


def _decorator_key(expr: ast.AST) -> str:
    """Stable key for decorator expressions for set-union logic."""
    try:
        return ast.dump(expr, include_attributes=False)
    except Exception:
        return repr(expr)


def _union_decorators(existing: List[ast.expr], new: List[ast.expr]) -> List[ast.expr]:
    """Return decorator list preserving existing order and appending missing from new."""
    result: List[ast.expr] = copy.deepcopy(existing)
    seen: Set[str] = { _decorator_key(d) for d in existing }
    for dec in new:
        k = _decorator_key(dec)
        if k not in seen:
            result.append(copy.deepcopy(dec))
            seen.add(k)
    return result


def _unified_diff(old: str, new: str, *, filename: str = "before.py", new_filename: str = "after.py") -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=filename, tofile=new_filename)
    return "".join(diff)


def merge_modules(existing_src: str, new_src: str) -> Tuple[str, List[str]]:
    """Merge new test module content into existing using structure-aware rules.

    Returns (merged_source_text, actions)
    """
    existing_mod = parse_module(existing_src)
    new_mod = parse_module(new_src)

    existing_idx = index_module(existing_mod)
    new_idx = index_module(new_mod)

    actions: List[str] = []

    # Start with a deep copy of existing module so we can mutate
    merged_mod: ast.Module = copy.deepcopy(existing_mod)

    # 1) Imports: append any missing normalized entries at the top (after existing imports)
    existing_imports: Set[ImportEntry] = existing_idx["imports"]  # type: ignore
    new_imports: Set[ImportEntry] = new_idx["imports"]  # type: ignore
    missing_imports = [e for e in new_imports if e not in existing_imports]
    if missing_imports:
        # Find insertion point: last position of an import statement in existing
        insert_pos = 0
        for i, node in enumerate(merged_mod.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_pos = i + 1
        for mi in missing_imports:
            merged_mod.body.insert(insert_pos, _import_entry_to_node(mi))
            insert_pos += 1
            actions.append(f"import:add:{mi.module or ''}:{mi.name}")

    # 2) Module-level functions
    existing_funcs: Dict[str, ast.AST] = existing_idx["functions"]  # type: ignore
    new_funcs: Dict[str, ast.AST] = new_idx["functions"]  # type: ignore
    # Build name->index mapping in merged_mod for fast replacement
    merged_func_positions: Dict[str, int] = {
        node.name: i
        for i, node in enumerate(merged_mod.body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name, new_fn in new_funcs.items():
        if name not in existing_funcs:
            merged_mod.body.append(copy.deepcopy(new_fn))
            actions.append(f"func:add:{name}")
        else:
            pos = merged_func_positions.get(name)
            if pos is not None:
                old_node = merged_mod.body[pos]
                if not ast_equal(old_node, new_fn):
                    replacement = copy.deepcopy(new_fn)
                    # Union decorators
                    if isinstance(old_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(replacement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        replacement.decorator_list = _union_decorators(old_node.decorator_list, replacement.decorator_list)
                    merged_mod.body[pos] = replacement
                    actions.append(f"func:update:{name}")

    # 3) Classes and their methods
    existing_classes: Dict[str, Tuple[ast.ClassDef, Dict[str, ast.AST]]] = existing_idx["classes"]  # type: ignore
    new_classes: Dict[str, Tuple[ast.ClassDef, Dict[str, ast.AST]]] = new_idx["classes"]  # type: ignore
    # Map class name -> position in merged_mod
    merged_class_positions: Dict[str, int] = {
        node.name: i
        for i, node in enumerate(merged_mod.body)
        if isinstance(node, ast.ClassDef)
    }
    for cls_name, (new_cls, new_methods) in new_classes.items():
        if cls_name not in existing_classes:
            merged_mod.body.append(copy.deepcopy(new_cls))
            actions.append(f"class:add:{cls_name}")
        else:
            # merge methods
            pos = merged_class_positions.get(cls_name)
            if pos is None:
                # Shouldn't happen, but if index missed it, append
                merged_mod.body.append(copy.deepcopy(new_cls))
                actions.append(f"class:add:{cls_name}")
                continue
            merged_cls: ast.ClassDef = merged_mod.body[pos]  # type: ignore
            # Map existing methods by name in merged class
            merged_methods_pos: Dict[str, int] = {
                n.name: j
                for j, n in enumerate(merged_cls.body)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for m_name, new_m in new_methods.items():
                if m_name not in merged_methods_pos:
                    merged_cls.body.append(copy.deepcopy(new_m))
                    actions.append(f"method:add:{cls_name}.{m_name}")
                else:
                    m_pos = merged_methods_pos[m_name]
                    old_m = merged_cls.body[m_pos]
                    if not ast_equal(old_m, new_m):
                        replacement_m = copy.deepcopy(new_m)
                        if isinstance(old_m, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(replacement_m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            replacement_m.decorator_list = _union_decorators(old_m.decorator_list, replacement_m.decorator_list)
                        merged_cls.body[m_pos] = replacement_m
                        actions.append(f"method:update:{cls_name}.{m_name}")

    # Unparse back to code
    merged_src = ast.unparse(merged_mod)
    # Ensure trailing newline
    if not merged_src.endswith("\n"):
        merged_src += "\n"
    return merged_src, actions


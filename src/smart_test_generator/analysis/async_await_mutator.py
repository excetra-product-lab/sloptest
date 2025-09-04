"""Async/await mutation operator for modern Python asynchronous code."""

import ast
import logging
from typing import Dict, List, Optional, Set, Any
import sys

from smart_test_generator.models.data_models import MutationType, Mutant
from smart_test_generator.analysis.mutation_engine import MutationOperator

logger = logging.getLogger(__name__)


class AsyncAwaitMutator(MutationOperator):
    """Mutate async/await patterns and asyncio primitives to test async edge cases."""
    
    def __init__(self):
        """Initialize async/await mutation mappings."""
        
        # Asyncio function mutations - common asyncio patterns to mutate
        self.asyncio_function_mutations = {
            'gather': ['wait', 'wait_for', 'create_task'],
            'wait': ['gather', 'wait_for', 'as_completed'],
            'wait_for': ['gather', 'wait', 'create_task'],
            'create_task': ['gather', 'ensure_future'],
            'ensure_future': ['create_task', 'gather'],
            'as_completed': ['wait', 'gather'],
            'run': ['run_until_complete', ''],  # asyncio.run() → removed
            'run_until_complete': ['run', ''],
            'sleep': ['', 'time.sleep'],  # asyncio.sleep → time.sleep or removed
        }
        
        # Async context manager mutations
        self.async_context_mutations = {
            'aenter': 'enter',      # __aenter__ → __enter__
            'aexit': 'exit',        # __aexit__ → __exit__
        }
        
        # Async iteration mutations
        self.async_iter_mutations = {
            'aiter': 'iter',        # __aiter__ → __iter__
            'anext': 'next',        # __anext__ → __next__
        }
        
        # Common async/await patterns to detect
        self.await_patterns = {
            'await_sleep',          # await asyncio.sleep()
            'await_gather',         # await asyncio.gather()
            'await_wait',           # await asyncio.wait()
            'await_task',           # await task
            'await_coroutine',      # await coroutine()
        }
        
        # Dangerous mutations that create coroutines instead of results
        self.dangerous_await_removals = {
            'create_task', 'gather', 'wait', 'wait_for', 'ensure_future'
        }
    
    def get_mutation_type(self) -> MutationType:
        """Get the mutation type for async/await mutations."""
        return MutationType.ASYNC_AWAIT
    
    def is_applicable(self, node: Any) -> bool:
        """Check if this operator can be applied to the given AST node."""
        return isinstance(node, (
            ast.AsyncFunctionDef, ast.FunctionDef, ast.Await, ast.AsyncWith, 
            ast.AsyncFor, ast.Call, ast.Attribute
        ))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate async/await mutants for the given source code."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            # Walk through all nodes to find async patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    mutants.extend(self._mutate_async_function_def(node, source_code))
                elif isinstance(node, ast.Await):
                    mutants.extend(self._mutate_await_expression(node, source_code))
                elif isinstance(node, ast.AsyncWith):
                    mutants.extend(self._mutate_async_with(node, source_code))
                elif isinstance(node, ast.AsyncFor):
                    mutants.extend(self._mutate_async_for(node, source_code))
                elif isinstance(node, ast.Call):
                    mutants.extend(self._mutate_asyncio_calls(node, source_code))
                elif isinstance(node, ast.FunctionDef):
                    mutants.extend(self._mutate_sync_to_async(node, source_code))
                    
        except Exception as e:
            logger.error(f"Error generating async/await mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_async_function_def(self, node: ast.AsyncFunctionDef, source_code: str) -> List[Mutant]:
        """Mutate async function definitions."""
        mutants = []
        
        # Mutation 1: async def → def (remove async keyword)
        mutant_id = f"ASYNC_DEF_REMOVE_{node.lineno}_{node.name}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code="async def",
            mutated_code="def",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 9,  # length of "async def"
            description=f"Remove async def from function: {node.name}",
            severity="critical",
            language="python"
        )
        mutants.append(mutant)
        
        # Mutation 2: Check for async special methods
        if node.name.startswith('__a') and node.name.endswith('__'):
            sync_method = self._get_sync_equivalent(node.name)
            if sync_method:
                mutant_id = f"ASYNC_METHOD_{node.lineno}_{node.name}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=node.name,
                    mutated_code=sync_method,
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(node.name)),
                    description=f"Convert async method to sync: {node.name} → {sync_method}",
                    severity="high",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_await_expression(self, node: ast.Await, source_code: str) -> List[Mutant]:
        """Mutate await expressions."""
        mutants = []
        
        # Get the awaited expression as string
        try:
            if sys.version_info >= (3, 9):
                awaited_expr = ast.unparse(node.value)
            else:
                awaited_expr = self._ast_to_string(node.value)
        except:
            awaited_expr = "<expression>"
        
        # Mutation 1: await expr → expr (remove await - dangerous!)
        mutant_id = f"AWAIT_REMOVE_{node.lineno}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code=f"await {awaited_expr}",
            mutated_code=awaited_expr,
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 5,  # length of "await"
            description=f"Remove await keyword (creates coroutine object instead of result)",
            severity="critical",
            language="python"
        )
        mutants.append(mutant)
        
        # Mutation 2: Specific asyncio function mutations
        if isinstance(node.value, ast.Call):
            mutants.extend(self._mutate_awaited_call(node, awaited_expr))
        
        # Mutation 3: await task → task.result() (for ast.Name nodes)
        if isinstance(node.value, ast.Name):
            mutant_id = f"AWAIT_TASK_RESULT_{node.lineno}"
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code=f"await {awaited_expr}",
                mutated_code=f"{awaited_expr}.result()",
                line_number=node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset',
                                 node.col_offset + len(f"await {awaited_expr}")),
                description="Replace await with .result() call (may raise if not done)",
                severity="high",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_awaited_call(self, await_node: ast.Await, awaited_expr: str) -> List[Mutant]:
        """Mutate specific awaited function calls."""
        mutants = []
        call_node = await_node.value
        
        # Check for asyncio.sleep() → time.sleep()
        if (isinstance(call_node.func, ast.Attribute) and 
            isinstance(call_node.func.value, ast.Name) and 
            call_node.func.value.id == 'asyncio' and 
            call_node.func.attr == 'sleep'):
            
            mutant_id = f"AWAIT_SLEEP_SYNC_{await_node.lineno}"
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code=awaited_expr,
                mutated_code="time.sleep" + awaited_expr[awaited_expr.find('('):],
                line_number=await_node.lineno,
                column_start=await_node.col_offset,
                column_end=getattr(await_node, 'end_col_offset', 
                                 await_node.col_offset + len(awaited_expr)),
                description="Replace asyncio.sleep with time.sleep (blocks event loop)",
                severity="critical",
                language="python"
            )
            mutants.append(mutant)
        
        # Check for task.result() equivalent mutations
        if (isinstance(call_node, ast.Name) or 
            (isinstance(call_node, ast.Call) and 
             self._is_task_creation_call(call_node))):
            
            # await task → task.result() (dangerous - may not be done)
            mutant_id = f"AWAIT_TASK_RESULT_{await_node.lineno}"
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code=f"await {awaited_expr}",
                mutated_code=f"{awaited_expr}.result()",
                line_number=await_node.lineno,
                column_start=await_node.col_offset,
                column_end=getattr(await_node, 'end_col_offset',
                                 await_node.col_offset + len(f"await {awaited_expr}")),
                description="Replace await with .result() call (may raise if not done)",
                severity="high",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_async_with(self, node: ast.AsyncWith, source_code: str) -> List[Mutant]:
        """Mutate async with statements."""
        mutants = []
        
        # Mutation: async with → with (remove async)
        mutant_id = f"ASYNC_WITH_REMOVE_{node.lineno}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code="async with",
            mutated_code="with",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 10,  # length of "async with"
            description="Remove async from with statement (may cause blocking)",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_async_for(self, node: ast.AsyncFor, source_code: str) -> List[Mutant]:
        """Mutate async for statements."""
        mutants = []
        
        # Mutation: async for → for (remove async)
        mutant_id = f"ASYNC_FOR_REMOVE_{node.lineno}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code="async for",
            mutated_code="for",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 9,  # length of "async for"
            description="Remove async from for loop (requires sync iterator)",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_asyncio_calls(self, node: ast.Call, source_code: str) -> List[Mutant]:
        """Mutate asyncio function calls."""
        mutants = []
        
        # Check for asyncio.function() patterns
        if (isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and 
            node.func.value.id == 'asyncio'):
            
            function_name = node.func.attr
            if function_name in self.asyncio_function_mutations:
                for substitute in self.asyncio_function_mutations[function_name]:
                    if substitute == '':  # Remove the call entirely
                        # Special handling for asyncio.run() - unwrap to just the inner call
                        if function_name == 'run' and node.args:
                            try:
                                if hasattr(node.args[0], 'id'):  # Simple name
                                    inner_call = node.args[0].id + "()"
                                else:
                                    import sys
                                    if sys.version_info >= (3, 9):
                                        inner_call = ast.unparse(node.args[0])
                                    else:
                                        inner_call = self._ast_to_string(node.args[0])
                                
                                mutant_id = f"ASYNCIO_RUN_UNWRAP_{node.lineno}"
                                mutant = Mutant(
                                    id=mutant_id,
                                    mutation_type=self.get_mutation_type(),
                                    original_code=f"asyncio.run({inner_call})",
                                    mutated_code=inner_call,
                                    line_number=node.lineno,
                                    column_start=node.col_offset,
                                    column_end=getattr(node, 'end_col_offset', 
                                                     node.col_offset + len(f"asyncio.run({inner_call})")),
                                    description="Remove asyncio.run wrapper, keeping inner call",
                                    severity="critical",
                                    language="python"
                                )
                                mutants.append(mutant)
                            except:
                                # Fallback to comment if unparsing fails
                                pass
                        else:
                            # General removal for other asyncio functions
                            mutant_id = f"ASYNCIO_REMOVE_{function_name}_{node.lineno}"
                            mutant = Mutant(
                                id=mutant_id,
                                mutation_type=self.get_mutation_type(),
                                original_code=f"asyncio.{function_name}",
                                mutated_code=f"# asyncio.{function_name} removed",
                                line_number=node.lineno,
                                column_start=node.col_offset,
                                column_end=getattr(node, 'end_col_offset', 
                                                 node.col_offset + len(f"asyncio.{function_name}")),
                                description=f"Remove asyncio.{function_name} call",
                                severity="critical",
                                language="python"
                            )
                            mutants.append(mutant)
                    else:  # Substitute with different asyncio function
                        original_call = f"asyncio.{function_name}"
                        mutated_call = f"asyncio.{substitute}" if substitute.startswith('asyncio.') else f"asyncio.{substitute}"
                        
                        mutant_id = f"ASYNCIO_SUB_{function_name}_{substitute}_{node.lineno}"
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=original_call,
                            mutated_code=mutated_call,
                            line_number=node.lineno,
                            column_start=node.func.col_offset,
                            column_end=getattr(node.func, 'end_col_offset',
                                             node.func.col_offset + len(original_call)),
                            description=f"Substitute asyncio function: {original_call} → {mutated_call}",
                            severity="medium",
                            language="python"
                        )
                        mutants.append(mutant)
        
        # Check for asyncio.run() specific mutations
        elif (isinstance(node.func, ast.Attribute) and 
              isinstance(node.func.value, ast.Name) and 
              node.func.value.id == 'asyncio' and 
              node.func.attr == 'run'):
            
            # asyncio.run(main()) → main() (remove event loop)
            if node.args:
                try:
                    if sys.version_info >= (3, 9):
                        inner_call = ast.unparse(node.args[0])
                    else:
                        inner_call = self._ast_to_string(node.args[0])
                    
                    mutant_id = f"ASYNCIO_RUN_REMOVE_{node.lineno}"
                    mutant = Mutant(
                        id=mutant_id,
                        mutation_type=self.get_mutation_type(),
                        original_code=f"asyncio.run({inner_call})",
                        mutated_code=inner_call,
                        line_number=node.lineno,
                        column_start=node.col_offset,
                        column_end=getattr(node, 'end_col_offset',
                                         node.col_offset + len(f"asyncio.run({inner_call})")),
                        description="Remove asyncio.run wrapper (returns coroutine instead of result)",
                        severity="critical",
                        language="python"
                    )
                    mutants.append(mutant)
                except:
                    pass  # Skip if unable to parse inner call
        
        return mutants
    
    def _mutate_sync_to_async(self, node: ast.FunctionDef, source_code: str) -> List[Mutant]:
        """Dangerously mutate sync functions to async (creates coroutines)."""
        mutants = []
        
        # Only add async to functions that might benefit from it (dangerous mutation)
        if (not node.name.startswith('_') and  # Skip private methods
            len(node.body) >= 1 and            # Include functions with at least one statement
            not self._has_async_calls_in_body(node.body)):  # Skip if already has async calls
            
            mutant_id = f"SYNC_TO_ASYNC_{node.lineno}_{node.name}"
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code="def",
                mutated_code="async def",
                line_number=node.lineno,
                column_start=node.col_offset,
                column_end=node.col_offset + 3,  # length of "def"
                description=f"Add async to function definition: {node.name} (dangerous - returns coroutine)",
                severity="critical",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _get_sync_equivalent(self, async_method_name: str) -> Optional[str]:
        """Get the synchronous equivalent of an async special method."""
        sync_equivalents = {
            '__aenter__': '__enter__',
            '__aexit__': '__exit__',
            '__aiter__': '__iter__',
            '__anext__': '__next__',
        }
        return sync_equivalents.get(async_method_name)
    
    def _is_task_creation_call(self, node: ast.Call) -> bool:
        """Check if a call creates an asyncio Task."""
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and 
                node.func.value.id == 'asyncio' and 
                node.func.attr in ['create_task', 'ensure_future']):
                return True
        return False
    
    def _has_async_calls_in_body(self, body: List[ast.stmt]) -> bool:
        """Check if function body contains async calls (await, async with, etc.)."""
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Await, ast.AsyncWith, ast.AsyncFor)):
                    return True
        return False
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation (Python < 3.9 compatibility)."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._ast_to_string(node.value)}.{node.attr}"
            elif isinstance(node, ast.Call):
                func_str = self._ast_to_string(node.func)
                args_str = ", ".join(self._ast_to_string(arg) for arg in node.args)
                return f"{func_str}({args_str})"
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            else:
                return f"<{type(node).__name__}>"
        except:
            return f"<unparseable:{type(node).__name__}>"

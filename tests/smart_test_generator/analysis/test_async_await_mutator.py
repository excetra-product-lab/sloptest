"""Tests for AsyncAwaitMutator - async/await mutations for modern Python."""

import ast
import pytest
from smart_test_generator.analysis.async_await_mutator import AsyncAwaitMutator
from smart_test_generator.models.data_models import MutationType


class TestAsyncAwaitMutator:
    """Test the AsyncAwaitMutator for async/await mutations."""
    
    @pytest.fixture
    def mutator(self):
        """Create an AsyncAwaitMutator instance."""
        return AsyncAwaitMutator()
    
    def test_get_mutation_type(self, mutator):
        """Test that the mutator returns correct mutation type."""
        assert mutator.get_mutation_type() == MutationType.ASYNC_AWAIT
    
    def test_is_applicable(self, mutator):
        """Test that mutator is applicable to correct node types."""
        # Should be applicable to async function definitions
        async_func_node = ast.parse("async def func(): pass").body[0]
        assert mutator.is_applicable(async_func_node)
        
        # Should be applicable to await expressions
        await_node = ast.parse("async def f(): await something()").body[0].body[0].value
        assert mutator.is_applicable(await_node)
        
        # Should be applicable to async with statements  
        async_with_node = ast.parse("async def f():\n    async with ctx: pass").body[0].body[0]
        assert mutator.is_applicable(async_with_node)
        
        # Should not be applicable to regular assignments
        assign_node = ast.parse("x = 5").body[0]
        assert not mutator.is_applicable(assign_node)


class TestAsyncFunctionDefMutations:
    """Test async function definition mutations."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_async_def_to_def_mutation(self, mutator):
        """Test mutation of async def to def."""
        source_code = '''
async def fetch_data():
    return await get_data()
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutation to remove async keyword
        async_def_mutants = [m for m in mutants if "async def" in m.original_code]
        assert len(async_def_mutants) >= 1
        
        # Check that it mutates to just "def"
        remove_async = [m for m in async_def_mutants if m.mutated_code == "def"]
        assert len(remove_async) >= 1
        
        # Should be critical severity
        assert any(m.severity == "critical" for m in async_def_mutants)
    
    def test_async_method_mutations(self, mutator):
        """Test mutations of async special methods."""
        source_code = '''
class AsyncResource:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def __aiter__(self):
        return self
    
    async def __anext__(self):
        raise StopAsyncIteration
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations for async special methods
        method_mutants = [m for m in mutants if "__a" in m.original_code and "__" in m.mutated_code]
        assert len(method_mutants) >= 4
        
        # Check specific method conversions
        method_conversions = {m.original_code: m.mutated_code for m in method_mutants}
        
        if "__aenter__" in method_conversions:
            assert method_conversions["__aenter__"] == "__enter__"
        if "__aexit__" in method_conversions:
            assert method_conversions["__aexit__"] == "__exit__"
        if "__aiter__" in method_conversions:
            assert method_conversions["__aiter__"] == "__iter__"
        if "__anext__" in method_conversions:
            assert method_conversions["__anext__"] == "__next__"
    
    def test_complex_async_function(self, mutator):
        """Test mutations of complex async functions."""
        source_code = '''
async def process_async_data(data: List[str]) -> Dict[str, Any]:
    async with aiofiles.open("file.txt") as f:
        content = await f.read()
    
    results = await asyncio.gather(*[
        process_item(item) async for item in data
    ])
    
    return {"content": content, "results": results}
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate multiple types of mutations
        assert len(mutants) >= 4
        
        # Should have async def mutation
        async_def_mutants = [m for m in mutants if "async def" in m.original_code]
        assert len(async_def_mutants) >= 1


class TestAwaitExpressionMutations:
    """Test await expression mutations."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_simple_await_removal(self, mutator):
        """Test removal of await keyword."""
        source_code = '''
async def fetch():
    result = await api_call()
    return result
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate await removal mutations
        await_mutants = [m for m in mutants if "await " in m.original_code and "await " not in m.mutated_code]
        assert len(await_mutants) >= 1
        
        # Should be critical severity (creates coroutine instead of result)
        assert any(m.severity == "critical" for m in await_mutants)
    
    def test_await_asyncio_sleep_mutation(self, mutator):
        """Test mutation of await asyncio.sleep() to time.sleep()."""
        source_code = '''
import asyncio

async def delayed_task():
    await asyncio.sleep(1.0)
    return "done"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate asyncio.sleep → time.sleep mutation
        sleep_mutants = [m for m in mutants if "asyncio.sleep" in m.original_code]
        assert len(sleep_mutants) >= 1
        
        # Check for time.sleep conversion
        time_sleep_mutants = [m for m in sleep_mutants if "time.sleep" in m.mutated_code]
        assert len(time_sleep_mutants) >= 1
        
        # Should be critical (blocks event loop)
        assert any(m.severity == "critical" for m in time_sleep_mutants)
    
    def test_await_task_result_mutation(self, mutator):
        """Test mutation of await task to task.result()."""
        source_code = '''
async def process():
    task = asyncio.create_task(some_coroutine())
    result = await task
    return result
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate await task → task.result() mutations
        task_mutants = [m for m in mutants if "await " in m.original_code and ".result()" in m.mutated_code]
        assert len(task_mutants) >= 1
        
        # Should be high severity
        assert any(m.severity in ["high", "critical"] for m in task_mutants)
    
    def test_complex_await_expressions(self, mutator):
        """Test mutations of complex await expressions."""
        source_code = '''
async def complex_awaits():
    # Multiple awaits in one line
    results = await asyncio.gather(
        fetch_user(1),
        fetch_user(2),
        fetch_user(3)
    )
    
    # Await in comprehension
    processed = [await process(item) for item in results]
    
    # Await with method chaining
    data = await client.get("/api").json()
    
    return processed, data
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate multiple await mutations
        await_mutants = [m for m in mutants if "await " in m.original_code]
        assert len(await_mutants) >= 3


class TestAsyncContextManagerMutations:
    """Test async context manager mutations."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_async_with_to_with_mutation(self, mutator):
        """Test mutation of async with to regular with."""
        source_code = '''
async def read_file():
    async with aiofiles.open("data.txt") as f:
        content = await f.read()
    return content
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate async with → with mutation
        async_with_mutants = [m for m in mutants if "async with" in m.original_code]
        assert len(async_with_mutants) >= 1
        
        # Check mutation to regular with
        with_mutants = [m for m in async_with_mutants if m.mutated_code == "with"]
        assert len(with_mutants) >= 1
        
        # Should be high severity
        assert any(m.severity == "high" for m in async_with_mutants)
    
    def test_nested_async_with(self, mutator):
        """Test mutations with nested async context managers."""
        source_code = '''
async def nested_contexts():
    async with database.transaction():
        async with file_lock("shared.txt"):
            async with aiofiles.open("data.txt") as f:
                return await f.read()
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations for all async with statements
        async_with_mutants = [m for m in mutants if "async with" in m.original_code]
        assert len(async_with_mutants) >= 3


class TestAsyncIterationMutations:
    """Test async iteration mutations."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_async_for_to_for_mutation(self, mutator):
        """Test mutation of async for to regular for."""
        source_code = '''
async def process_stream():
    results = []
    async for item in stream:
        results.append(await process(item))
    return results
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate async for → for mutation
        async_for_mutants = [m for m in mutants if "async for" in m.original_code]
        assert len(async_for_mutants) >= 1
        
        # Check mutation to regular for
        for_mutants = [m for m in async_for_mutants if m.mutated_code == "for"]
        assert len(for_mutants) >= 1
        
        # Should be high severity
        assert any(m.severity == "high" for m in async_for_mutants)
    
    def test_async_comprehensions(self, mutator):
        """Test handling of async comprehensions."""
        source_code = '''
async def async_comprehensions():
    # Async list comprehension
    results = [await process(x) async for x in stream]
    
    # Async generator expression  
    generator = (await transform(x) async for x in data)
    
    return results, generator
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should handle async comprehensions (may generate await mutations)
        assert len(mutants) >= 2


class TestAsyncioFunctionMutations:
    """Test mutations of asyncio function calls."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_asyncio_gather_mutations(self, mutator):
        """Test mutations of asyncio.gather() calls."""
        source_code = '''
async def fetch_multiple():
    results = await asyncio.gather(
        fetch_user(1),
        fetch_user(2),
        fetch_user(3)
    )
    return results
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate asyncio.gather mutations
        gather_mutants = [m for m in mutants if "asyncio.gather" in m.original_code]
        assert len(gather_mutants) >= 1
        
        # Check for substitutions to other asyncio functions
        substitutions = [m.mutated_code for m in gather_mutants]
        assert any("wait" in sub or "wait_for" in sub for sub in substitutions)
    
    def test_asyncio_wait_mutations(self, mutator):
        """Test mutations of asyncio.wait() calls."""
        source_code = '''
async def wait_for_tasks():
    done, pending = await asyncio.wait([
        task1, task2, task3
    ], timeout=5.0)
    return done, pending
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate asyncio.wait mutations
        wait_mutants = [m for m in mutants if "asyncio.wait" in m.original_code]
        assert len(wait_mutants) >= 1
    
    def test_asyncio_run_mutation(self, mutator):
        """Test mutations of asyncio.run() calls."""
        source_code = '''
def main():
    result = asyncio.run(async_main())
    return result
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate asyncio.run removal mutation
        run_mutants = [m for m in mutants if "asyncio.run" in m.original_code]
        assert len(run_mutants) >= 1
        
        # Should remove the asyncio.run wrapper
        inner_call_mutants = [m for m in run_mutants if "async_main()" in m.mutated_code]
        assert len(inner_call_mutants) >= 1
        
        # Should be critical (returns coroutine instead of result)
        assert any(m.severity == "critical" for m in run_mutants)


class TestSyncToAsyncMutations:
    """Test dangerous mutations from sync to async functions."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_sync_to_async_mutation(self, mutator):
        """Test dangerous mutation of def to async def."""
        source_code = '''
def calculate_result(x, y):
    # Multi-line function without async calls
    result = x + y
    processed = result * 2
    return processed
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate sync to async mutations (dangerous)
        sync_to_async = [m for m in mutants if m.original_code == "def" and "async" in m.mutated_code]
        assert len(sync_to_async) >= 1
        
        # Should be critical severity
        assert any(m.severity == "critical" for m in sync_to_async)
    
    def test_skip_private_methods(self, mutator):
        """Test that private methods are skipped for sync-to-async mutations."""
        source_code = '''
def _private_method():
    return "private"

def __dunder_method__(self):
    return "dunder"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should not generate sync-to-async for private methods
        sync_to_async = [m for m in mutants if m.original_code == "def" and "async" in m.mutated_code]
        assert len(sync_to_async) == 0
    
    def test_skip_functions_with_async_calls(self, mutator):
        """Test that functions with async calls are skipped for sync-to-async."""
        source_code = '''
async def mixed_function():
    regular_call()
    result = await async_call()
    return result
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should not add async to functions that already have async calls
        sync_to_async = [m for m in mutants if "def" == m.original_code and "async def" == m.mutated_code]
        # Function is already async, so no sync-to-async mutations should occur
        assert len(sync_to_async) == 0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in async/await mutations."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_invalid_async_syntax_handling(self, mutator):
        """Test handling of invalid async syntax."""
        invalid_code = '''
async def broken():
    await invalid_syntax_here
'''
        # Should not raise an exception
        mutants = mutator.generate_mutants(invalid_code, "test.py")
        assert isinstance(mutants, list)
    
    def test_mixed_sync_async_code(self, mutator):
        """Test handling of files with mixed sync and async code."""
        source_code = '''
def sync_function():
    return "sync"

async def async_function():
    return await get_data()

class MixedClass:
    def sync_method(self):
        return "sync"
    
    async def async_method(self):
        return await self.get_async_data()
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should handle mixed code without crashing
        assert len(mutants) >= 3
        
        # Should have both sync-to-async and async-to-sync mutations
        sync_to_async = [m for m in mutants if m.original_code == "def" and "async" in m.mutated_code]
        async_to_sync = [m for m in mutants if "async def" in m.original_code and m.mutated_code == "def"]
        
        assert len(sync_to_async) >= 1
        assert len(async_to_sync) >= 1
    
    def test_no_async_code_file(self, mutator):
        """Test file with no async code."""
        source_code = '''
def regular_function():
    return "regular"

class RegularClass:
    def method(self):
        return "method"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate sync-to-async mutations or return empty list
        assert isinstance(mutants, list)
        
        # If mutations are generated, they should be sync-to-async
        for mutant in mutants:
            assert mutant.mutation_type == MutationType.ASYNC_AWAIT


class TestMutantQuality:
    """Test the quality and properties of generated mutants."""
    
    @pytest.fixture
    def mutator(self):
        return AsyncAwaitMutator()
    
    def test_mutant_ids_are_unique(self, mutator):
        """Test that generated mutants have unique IDs."""
        source_code = '''
async def complex_async():
    result = await asyncio.gather(fetch_a(), fetch_b())
    async with resource():
        async for item in stream:
            await process(item)
    return result
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        mutant_ids = [m.id for m in mutants]
        assert len(mutant_ids) == len(set(mutant_ids))
    
    def test_severity_levels_appropriate(self, mutator):
        """Test that mutants have appropriate severity levels."""
        source_code = '''
async def severity_test():
    await asyncio.sleep(1)  # Critical when removed
    async with resource():  # High severity
        pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        severities = [m.severity for m in mutants]
        assert len(set(severities)) >= 2
        
        # Should have high severity mutations
        assert any(sev in ["high", "critical"] for sev in severities)
    
    def test_descriptions_are_informative(self, mutator):
        """Test that descriptions explain the async implications."""
        source_code = '''
async def test_descriptions():
    await some_coroutine()
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        for mutant in mutants:
            assert len(mutant.description) > 10
            # Should mention async-related concepts
            description_lower = mutant.description.lower()
            assert any(term in description_lower for term in [
                "async", "await", "coroutine", "event loop", "block"
            ])
    
    def test_line_and_column_information(self, mutator):
        """Test that mutants have correct position information."""
        source_code = '''async def test():
    await call()
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        for mutant in mutants:
            assert mutant.line_number >= 1
            assert mutant.column_start >= 0
            assert mutant.column_end > mutant.column_start
            assert mutant.language == "python"

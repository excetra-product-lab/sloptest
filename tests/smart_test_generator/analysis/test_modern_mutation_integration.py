"""Integration tests for modern Python mutators working together."""

import pytest
from smart_test_generator.analysis.mutation_engine import MutationTestingEngine
from smart_test_generator.analysis.type_hint_mutator import TypeHintMutator
from smart_test_generator.analysis.async_await_mutator import AsyncAwaitMutator
from smart_test_generator.analysis.dataclass_mutator import DataclassMutator
from smart_test_generator.models.data_models import MutationType


class TestModernMutatorIntegration:
    """Test integration of all modern Python mutators."""
    
    def test_all_modern_mutators_available(self):
        """Test that all modern mutators can be imported and instantiated."""
        type_mutator = TypeHintMutator()
        async_mutator = AsyncAwaitMutator()
        dataclass_mutator = DataclassMutator()
        
        assert type_mutator.get_mutation_type() == MutationType.TYPE_HINT
        assert async_mutator.get_mutation_type() == MutationType.ASYNC_AWAIT
        assert dataclass_mutator.get_mutation_type() == MutationType.DATACLASS
    
    def test_mutation_engine_includes_modern_mutators(self):
        """Test that MutationTestingEngine includes modern mutators by default."""
        engine = MutationTestingEngine()
        
        # Should include modern mutators in operators list
        mutator_types = [op.get_mutation_type() for op in engine.operators]
        
        assert MutationType.TYPE_HINT in mutator_types
        assert MutationType.ASYNC_AWAIT in mutator_types
        assert MutationType.DATACLASS in mutator_types
    
    def test_complex_modern_python_file(self):
        """Test mutations on a complex file using all modern Python features."""
        source_code = '''
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
import asyncio
from abc import ABC, abstractmethod

@dataclass
class UserData:
    """User data with modern Python features."""
    name: str
    age: Optional[int] = None
    emails: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.age is not None and self.age < 0:
            raise ValueError("Age cannot be negative")

class UserService(ABC):
    """Abstract user service with async methods."""
    
    @abstractmethod
    async def get_user(self, user_id: int) -> Optional[UserData]:
        pass
    
    @abstractmethod
    async def save_user(self, user: UserData) -> bool:
        pass

class DatabaseUserService(UserService):
    """Database implementation of user service."""
    
    def __init__(self, connection_pool: Any):
        self.pool = connection_pool
    
    async def get_user(self, user_id: int) -> Optional[UserData]:
        """Get user by ID with type hints and async."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                "SELECT name, age, emails, metadata FROM users WHERE id = $1",
                user_id
            )
            
            if result:
                return UserData(
                    name=result['name'],
                    age=result['age'],
                    emails=result['emails'] or [],
                    metadata=result['metadata'] or {}
                )
            return None
    
    async def save_user(self, user: UserData) -> bool:
        """Save user with async operations."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO users (name, age, emails, metadata) VALUES ($1, $2, $3, $4)",
                    user.name, user.age, user.emails, user.metadata
                )
                return True
        except Exception:
            return False
    
    async def get_multiple_users(
        self, 
        user_ids: List[int]
    ) -> Dict[int, Optional[UserData]]:
        """Get multiple users concurrently."""
        tasks = [self.get_user(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks)
        
        return dict(zip(user_ids, results))

@dataclass(frozen=True, order=True)
class ImmutableUserSummary:
    """Immutable user summary with ordering."""
    user_count: int
    active_users: int
    average_age: Optional[float] = None
    top_domains: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.user_count < 0:
            object.__setattr__(self, 'user_count', 0)

async def process_users_batch(
    service: UserService,
    user_ids: List[int],
    batch_size: int = 10
) -> Union[List[UserData], None]:
    """Process users in batches with modern Python features."""
    if not user_ids:
        return None
    
    all_users: List[UserData] = []
    
    # Process in batches
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i + batch_size]
        
        # Use asyncio.gather for concurrent processing
        tasks = [service.get_user(uid) for uid in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_users = [
            user for user in batch_results 
            if isinstance(user, UserData)
        ]
        all_users.extend(valid_users)
        
        # Small delay between batches
        await asyncio.sleep(0.1)
    
    return all_users if all_users else None
'''
        
        # Test with individual mutators
        type_mutator = TypeHintMutator()
        async_mutator = AsyncAwaitMutator()
        dataclass_mutator = DataclassMutator()
        
        type_mutants = type_mutator.generate_mutants(source_code, "complex_test.py")
        async_mutants = async_mutator.generate_mutants(source_code, "complex_test.py")
        dataclass_mutants = dataclass_mutator.generate_mutants(source_code, "complex_test.py")
        
        # Should generate mutations from all mutators
        assert len(type_mutants) >= 10  # Many type hints to mutate
        assert len(async_mutants) >= 8   # Many async/await patterns
        assert len(dataclass_mutants) >= 6  # Two dataclasses with various features
        
        # Test with mutation engine (all together)
        engine = MutationTestingEngine()
        all_mutants = engine.generate_mutants_from_code(source_code, "complex_test.py")
        
        # Should include mutations from all mutator types
        mutation_types = [m.mutation_type for m in all_mutants]
        assert MutationType.TYPE_HINT in mutation_types
        assert MutationType.ASYNC_AWAIT in mutation_types
        assert MutationType.DATACLASS in mutation_types
        
        # Total should be sum of individual mutators (approximately)
        individual_total = len(type_mutants) + len(async_mutants) + len(dataclass_mutants)
        # Allow for some variance due to integration effects
        assert len(all_mutants) >= individual_total * 0.8
    
    def test_mutator_configuration(self):
        """Test that mutators can be configured via config."""
        # Test with all mutators enabled
        config_all_enabled = {
            'quality': {
                'modern_mutators': {
                    'enable_type_hints': True,
                    'enable_async_await': True,
                    'enable_dataclass': True
                }
            }
        }
        
        engine_all = MutationTestingEngine(config=config_all_enabled)
        mutator_types_all = [op.get_mutation_type() for op in engine_all.operators]
        
        assert MutationType.TYPE_HINT in mutator_types_all
        assert MutationType.ASYNC_AWAIT in mutator_types_all
        assert MutationType.DATACLASS in mutator_types_all
        
        # Test with only type hints enabled
        config_type_only = {
            'quality': {
                'modern_mutators': {
                    'enable_type_hints': True,
                    'enable_async_await': False,
                    'enable_dataclass': False
                }
            }
        }
        
        engine_type_only = MutationTestingEngine(config=config_type_only)
        mutator_types_type_only = [op.get_mutation_type() for op in engine_type_only.operators]
        
        assert MutationType.TYPE_HINT in mutator_types_type_only
        assert MutationType.ASYNC_AWAIT not in mutator_types_type_only
        assert MutationType.DATACLASS not in mutator_types_type_only


class TestModernPythonVersionCompatibility:
    """Test compatibility with different Python versions and features."""
    
    def test_python_310_features(self):
        """Test handling of Python 3.10+ features."""
        # Union syntax (X | Y) - Python 3.10+
        source_code_310 = '''
def func(x: int | str) -> bool | None:
    return True

@dataclass(slots=True, kw_only=True)
class ModernData:
    name: str
    value: int | float = 0
'''
        
        type_mutator = TypeHintMutator()
        dataclass_mutator = DataclassMutator()
        
        # Should handle gracefully even on older Python
        type_mutants = type_mutator.generate_mutants(source_code_310, "python310_test.py")
        dataclass_mutants = dataclass_mutator.generate_mutants(source_code_310, "python310_test.py")
        
        # Should not crash
        assert isinstance(type_mutants, list)
        assert isinstance(dataclass_mutants, list)
    
    def test_python_39_features(self):
        """Test handling of Python 3.9+ features."""
        source_code_39 = '''
from typing import Optional, List, Dict

def process(items: list[str]) -> dict[str, int]:  # Python 3.9+ generic syntax
    return {item: len(item) for item in items}

@dataclass
class GenericData:
    items: list[str] = field(default_factory=list)
    mapping: dict[str, int] = field(default_factory=dict)
'''
        
        type_mutator = TypeHintMutator()
        dataclass_mutator = DataclassMutator()
        
        type_mutants = type_mutator.generate_mutants(source_code_39, "python39_test.py")
        dataclass_mutants = dataclass_mutator.generate_mutants(source_code_39, "python39_test.py")
        
        # Should handle gracefully
        assert isinstance(type_mutants, list)
        assert isinstance(dataclass_mutants, list)


class TestErrorHandlingAndEdgeCases:
    """Test error handling across all modern mutators."""
    
    def test_invalid_syntax_handling(self):
        """Test that invalid syntax is handled gracefully by all mutators."""
        invalid_code = '''
@dataclass(invalid_param=???)
class BrokenClass:
    field: invalid_type_syntax = await broken_await
    
async def broken_async(x: Union[???]) -> ???:
    await invalid_syntax_here
'''
        
        engine = MutationTestingEngine()
        
        # Should not raise exceptions
        try:
            mutants = engine.generate_mutants_from_code(invalid_code, "broken_test.py")
            assert isinstance(mutants, list)  # May be empty, but should be a list
        except Exception as e:
            pytest.fail(f"Should handle invalid syntax gracefully, but raised: {e}")
    
    def test_empty_file_handling(self):
        """Test handling of empty or minimal files."""
        empty_code = ""
        minimal_code = "pass"
        
        engine = MutationTestingEngine()
        
        empty_mutants = engine.generate_mutants_from_code(empty_code, "empty.py")
        minimal_mutants = engine.generate_mutants_from_code(minimal_code, "minimal.py")
        
        assert isinstance(empty_mutants, list)
        assert isinstance(minimal_mutants, list)
        assert len(empty_mutants) == 0
        assert len(minimal_mutants) == 0
    
    def test_large_file_performance(self):
        """Test performance with moderately large files."""
        # Generate a moderately complex file programmatically
        large_code = '''
from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass, field
import asyncio

'''
        
        # Add multiple classes and functions
        for i in range(5):
            large_code += f'''
@dataclass
class DataClass{i}:
    name: str = "default_{i}"
    value: Optional[int] = None
    items: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.value is None:
            self.value = {i}

async def async_function_{i}(
    data: Union[str, int], 
    optional_param: Optional[Dict[str, Any]] = None
) -> List[DataClass{i}]:
    results = []
    
    async with some_context():
        for j in range({i + 1}):
            await asyncio.sleep(0.01)
            item = DataClass{i}(name=f"item_{{j}}", value=j)
            results.append(item)
    
    return results
'''
        
        engine = MutationTestingEngine()
        
        # Should handle moderately large files efficiently
        import time
        start_time = time.time()
        mutants = engine.generate_mutants_from_code(large_code, "large_test.py")
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0  # 10 seconds max
        assert isinstance(mutants, list)
        assert len(mutants) >= 20  # Should generate many mutations


class TestMutationQualityIntegration:
    """Test the quality of mutations across all modern mutators."""
    
    def test_no_duplicate_mutations(self):
        """Test that different mutators don't generate duplicate mutations."""
        source_code = '''
@dataclass
class TestData:
    name: Optional[str] = None
    
async def process_data(data: TestData) -> Union[str, None]:
    if data.name:
        return await transform_name(data.name)
    return None
'''
        
        engine = MutationTestingEngine()
        mutants = engine.generate_mutants_from_code(source_code, "test.py")
        
        # Check for duplicate mutant IDs
        mutant_ids = [m.id for m in mutants]
        assert len(mutant_ids) == len(set(mutant_ids))
        
        # Check for duplicate mutations (same line, same change)
        mutation_signatures = [(m.line_number, m.column_start, m.original_code, m.mutated_code) for m in mutants]
        assert len(mutation_signatures) == len(set(mutation_signatures))
    
    def test_severity_distribution(self):
        """Test that mutations have appropriate severity distribution."""
        source_code = '''
@dataclass
class CriticalData:
    items: List[str] = field(default_factory=list)  # Critical if changed to default=[]
    
async def critical_async() -> Optional[str]:
    result = await asyncio.gather(task1(), task2())  # Critical if await removed
    return result[0] if result else None
'''
        
        engine = MutationTestingEngine()
        mutants = engine.generate_mutants_from_code(source_code, "severity_test.py")
        
        severities = [m.severity for m in mutants]
        
        # Should have mix of severities
        assert len(set(severities)) >= 2
        
        # Should have some critical mutations
        assert "critical" in severities
        
        # All severities should be valid
        valid_severities = {"low", "medium", "high", "critical"}
        assert all(sev in valid_severities for sev in severities)
    
    def test_informative_descriptions(self):
        """Test that all modern mutators generate informative descriptions."""
        source_code = '''
@dataclass(frozen=True)
class User:
    name: str
    age: Optional[int] = None
    
async def get_user() -> User:
    await asyncio.sleep(1)
    return User(name="test", age=25)
'''
        
        engine = MutationTestingEngine()
        mutants = engine.generate_mutants_from_code(source_code, "description_test.py")
        
        # All descriptions should be informative
        for mutant in mutants:
            assert len(mutant.description) > 15  # Reasonable minimum length
            
            # Descriptions should either contain the original code or be descriptive
            # Some generic descriptions like "Remove await keyword" are acceptable
            contains_original = mutant.original_code in mutant.description
            contains_keywords = any(keyword in mutant.description.lower() 
                                  for keyword in ['remove', 'change', 'mutate', 'replace', 'add'])
            assert contains_original or contains_keywords, f"Description should be informative: {mutant.description}"
            
            # For non-removal mutations, mutated code should be in description OR it should be descriptive
            if not (mutant.mutated_code.startswith("# ") and "removed" in mutant.mutated_code):
                contains_mutated = mutant.mutated_code in mutant.description
                is_descriptive = any(keyword in mutant.description.lower() 
                                   for keyword in ['remove', 'change', 'mutate', 'replace', 'add', 'keyword'])
                assert contains_mutated or is_descriptive, f"Description should mention change: {mutant.description}"
            
            # Should contain mutation type context
            description_lower = mutant.description.lower()
            if mutant.mutation_type == MutationType.TYPE_HINT:
                assert any(term in description_lower for term in ["type", "hint", "optional", "union"])
            elif mutant.mutation_type == MutationType.ASYNC_AWAIT:
                assert any(term in description_lower for term in ["async", "await", "coroutine"])
            elif mutant.mutation_type == MutationType.DATACLASS:
                assert any(term in description_lower for term in ["dataclass", "field", "frozen"])


# Add a method to MutationTestingEngine for testing
def generate_mutants_from_code(self, source_code: str, filename: str):
    """Generate mutants directly from source code string."""
    mutants = []
    for operator in self.operators:
        operator_mutants = operator.generate_mutants(source_code, filename)
        mutants.extend(operator_mutants)
    
    # Remove duplicates and sort
    unique_mutants = self._deduplicate_mutants(mutants) 
    return sorted(unique_mutants, key=lambda m: (m.line_number, m.column_start))

# Monkey patch for testing
MutationTestingEngine.generate_mutants_from_code = generate_mutants_from_code

# Modern Python Mutation Testing

This guide covers the advanced mutation operators for modern Python features introduced in Task 6, including type hints, async/await patterns, and dataclass mutations.

## Overview

The Smart Test Generator now includes three sophisticated mutation operators that target modern Python patterns:

- **Type Hint Mutations** - Test type annotation robustness
- **Async/Await Mutations** - Detect concurrency and asynchronous code issues  
- **Dataclass Mutations** - Validate dataclass behavior and configuration

These mutators help identify test gaps that traditional mutation testing misses in contemporary Python codebases.

## Type Hint Mutations

### What It Tests

Type hint mutations test whether your tests actually validate the type constraints you've declared. Many tests pass with incorrect types due to Python's dynamic nature.

### Mutation Categories

#### Optional Type Mutations

**Code:**
```python
def get_user_name(user_id: Optional[int]) -> Optional[str]:
    if user_id is None:
        return None
    return f"user_{user_id}"
```

**Mutations Generated:**
- `Optional[int]` → `int` (removes None handling)  
- `Optional[int]` → `None` (forces None-only input)
- `Optional[str]` → `str` (removes None return possibility)
- `Optional[str]` → `None` (forces None-only return)

**Test Impact:**
```python
# This test might pass with mutations if not comprehensive
def test_get_user_name():
    assert get_user_name(123) == "user_123"  # Missing None test cases!

# Better test that catches mutations
def test_get_user_name_comprehensive():
    # Test normal case
    assert get_user_name(123) == "user_123"
    # Test None input (catches Optional[int] → int mutation)
    assert get_user_name(None) is None
    # Test return type handling (catches Optional[str] mutations)
    result = get_user_name(456)
    assert result is None or isinstance(result, str)
```

#### Union Type Mutations

**Code:**
```python
def process_id(identifier: Union[int, str]) -> Union[bool, str]:
    if isinstance(identifier, int):
        return identifier > 0
    return f"processed_{identifier}"
```

**Mutations Generated:**
- `Union[int, str]` → `int` (removes string handling)
- `Union[int, str]` → `str` (removes integer handling)  
- `Union[bool, str]` → `bool` (removes string return)
- `Union[bool, str]` → `str` (removes boolean return)

**Test Impact:**
```python
def test_process_id_comprehensive():
    # Test both input types (catches Union input mutations)
    assert process_id(123) is True
    assert process_id(-5) is False
    assert process_id("test") == "processed_test"
    
    # Test both return types (catches Union return mutations)
    int_result = process_id(42)
    str_result = process_id("data")
    assert isinstance(int_result, bool)
    assert isinstance(str_result, str)
```

#### Generic Type Mutations

**Code:**
```python
def merge_data(items: List[Dict[str, int]]) -> Dict[str, List[int]]:
    result = {}
    for item in items:
        for key, value in item.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    return result
```

**Mutations Generated:**
- `List[Dict[str, int]]` → `list` (removes type parameter validation)
- `List[Dict[str, int]]` → `[]` (changes to empty list default)
- `Dict[str, List[int]]` → `dict` (removes return type constraints)

### Configuration

```yaml
quality:
  modern_mutators:
    enable_type_hints: true
    type_hints_severity: 'medium'  # low, medium, high
```

## Async/Await Mutations

### What It Tests  

Async mutations test whether your tests properly handle:
- Coroutines vs actual results
- Event loop blocking operations
- Proper async resource cleanup
- Concurrency patterns

### Mutation Categories

#### Async Function Definition Mutations

**Code:**
```python
async def fetch_user_data(user_id: int) -> UserData:
    async with database.connection() as conn:
        result = await conn.execute("SELECT * FROM users WHERE id = ?", user_id)
        return UserData.from_row(result)
```

**Mutations Generated:**
- `async def fetch_user_data` → `def fetch_user_data` (returns coroutine instead of result)

**Test Impact:**
```python
# This test will fail with async def → def mutation
@pytest.mark.asyncio
async def test_fetch_user_data():
    user = await fetch_user_data(123)
    assert isinstance(user, UserData)
    assert user.id == 123

# This test catches the mutation
@pytest.mark.asyncio  
async def test_fetch_user_data_is_coroutine():
    # Ensure it returns actual data, not a coroutine
    user = await fetch_user_data(123)
    assert not asyncio.iscoroutine(user)  # Catches async → sync mutation
    assert isinstance(user, UserData)
```

#### Await Expression Mutations

**Code:**
```python
async def process_batch(items: List[str]) -> List[Result]:
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

**Mutations Generated:**
- `await asyncio.gather(*tasks)` → `asyncio.gather(*tasks)` (returns coroutine list instead of results)
- `await asyncio.gather(...)` → `asyncio.wait(...)` (changes concurrency pattern)

**Test Impact:**
```python
@pytest.mark.asyncio
async def test_process_batch():
    items = ["item1", "item2", "item3"]
    results = await process_batch(items)
    
    # Catches await removal mutation
    assert not any(asyncio.iscoroutine(result) for result in results)
    assert len(results) == 3
    assert all(isinstance(r, Result) for r in results)
```

#### Asyncio Function Mutations

**Code:**
```python
async def delayed_operation():
    await asyncio.sleep(1.0)  # Async delay
    return "completed"
```

**Mutations Generated:**
- `await asyncio.sleep(1.0)` → `time.sleep(1.0)` (blocks event loop!)

**Test Impact:**
```python
@pytest.mark.asyncio
async def test_delayed_operation_non_blocking():
    import time
    start_time = time.time()
    
    # Run multiple concurrent operations
    tasks = [delayed_operation() for _ in range(3)]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    # Should complete concurrently in ~1 second, not 3 seconds
    # Catches asyncio.sleep → time.sleep mutation
    assert end_time - start_time < 2.0  
    assert len(results) == 3
    assert all(result == "completed" for result in results)
```

### Configuration

```yaml
quality:
  modern_mutators:
    enable_async_await: true
    async_severity: 'high'  # Async bugs often critical
```

## Dataclass Mutations

### What It Tests

Dataclass mutations test whether your tests validate:
- Immutability constraints (frozen=True)
- Field initialization behavior
- Default value sharing (mutable defaults)
- Post-initialization logic

### Mutation Categories

#### Decorator Parameter Mutations

**Code:**
```python
@dataclass(frozen=True, order=True)
class ImmutablePoint:
    x: float
    y: float
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
```

**Mutations Generated:**
- `@dataclass(frozen=True, order=True)` → `@dataclass(frozen=False, order=True)` (removes immutability)
- `frozen=True` → `frozen=False` (allows modification)
- `order=True` → `order=False` (disables comparison operators)
- `@dataclass(...)` → `# @dataclass removed` (removes entire decorator)

**Test Impact:**
```python
def test_immutable_point():
    p1 = ImmutablePoint(1.0, 2.0)
    p2 = ImmutablePoint(2.0, 3.0)
    
    # Test immutability (catches frozen=False mutation)
    with pytest.raises(FrozenInstanceError):
        p1.x = 5.0
    
    # Test ordering (catches order=False mutation)  
    assert p1 < p2  # Will fail if order=False
    
    # Test it's still a dataclass (catches decorator removal)
    assert hasattr(p1, '__dataclass_fields__')
```

#### Dangerous Field Mutations

**Code:**
```python
@dataclass
class UserPreferences:
    tags: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def add_tag(self, tag: str):
        self.tags.append(tag)
```

**Mutations Generated:**
- `field(default_factory=list)` → `field(default=[])` (DANGEROUS: shared mutable default!)
- `field(default_factory=dict)` → `field(default={})` (DANGEROUS: shared mutable default!)

**Test Impact:**
```python
def test_user_preferences_no_shared_mutables():
    # Create two instances
    user1 = UserPreferences()
    user2 = UserPreferences()
    
    # Modify one instance
    user1.add_tag("python")
    user1.settings["theme"] = "dark"
    
    # Other instance should be unaffected
    # This catches the dangerous default_factory → default mutations!
    assert len(user2.tags) == 0  # Fails with shared mutable default
    assert len(user2.settings) == 0  # Fails with shared mutable default
    
    # Instances should have separate mutable objects
    assert user1.tags is not user2.tags
    assert user1.settings is not user2.settings
```

#### Post-Init Mutations

**Code:**
```python
@dataclass
class ValidatedUser:
    name: str
    email: str
    normalized_email: str = field(init=False)
    
    def __post_init__(self):
        if not self.email or "@" not in self.email:
            raise ValueError("Invalid email address")
        self.normalized_email = self.email.lower().strip()
```

**Mutations Generated:**
- `def __post_init__(self):` → `# def __post_init__ removed`

**Test Impact:**
```python
def test_validated_user_post_init():
    # Test normal case
    user = ValidatedUser("John", "John@Example.COM")
    assert user.normalized_email == "john@example.com"
    
    # Test validation (catches __post_init__ removal)
    with pytest.raises(ValueError):
        ValidatedUser("Jane", "invalid-email")
    
    # Test computed field is set (catches __post_init__ removal)  
    user2 = ValidatedUser("Alice", "ALICE@test.ORG")
    assert hasattr(user2, 'normalized_email')
    assert user2.normalized_email == "alice@test.org"
```

### Configuration

```yaml
quality:
  modern_mutators:
    enable_dataclass: true
    dataclass_severity: 'medium'
```

## Configuration Examples

### Enable All Modern Mutators
```yaml
quality:
  enable_mutation_testing: true
  modern_mutators:
    enable_type_hints: true
    enable_async_await: true
    enable_dataclass: true
    type_hints_severity: 'medium'
    async_severity: 'high'
    dataclass_severity: 'medium'
```

### Type Hints Only
```yaml
quality:
  modern_mutators:
    enable_type_hints: true
    enable_async_await: false
    enable_dataclass: false
```

### Performance Tuning
```yaml
quality:
  max_mutants_per_file: 100  # Increase for comprehensive testing
  mutation_timeout: 60       # Increase for complex async tests
  modern_mutators:
    enable_type_hints: true
    enable_async_await: true
    enable_dataclass: true
```

## Best Practices

### Writing Tests That Catch Mutations

1. **Type Hints**: Test both success and failure cases for each type variant
2. **Async/Await**: Always test that results are actual values, not coroutines
3. **Dataclass**: Test immutability, field behavior, and shared state scenarios

### Test Examples for High Mutation Score

```python
# Example: Comprehensive async test
@pytest.mark.asyncio
async def test_async_comprehensive():
    # Test normal operation
    result = await async_function("input")
    
    # Test result is not a coroutine (catches async removal)
    assert not asyncio.iscoroutine(result)
    
    # Test concurrent behavior (catches concurrency mutations)
    start_time = time.time()
    tasks = [async_function(f"input_{i}") for i in range(3)]
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    assert duration < 2.0  # Should be concurrent, not sequential
    assert len(results) == 3

# Example: Comprehensive type hint test
def test_union_types_comprehensive():
    # Test all Union branches
    assert process_data("string") == "processed_string"  # str branch
    assert process_data(42) == 84                        # int branch
    
    # Test Optional None case
    assert process_optional(None) is None
    assert process_optional("value") == "processed_value"
    
    # Test return type validation
    str_result = process_data("test")
    int_result = process_data(100)
    assert isinstance(str_result, str)
    assert isinstance(int_result, int)

# Example: Comprehensive dataclass test  
def test_dataclass_comprehensive():
    # Test normal construction
    obj1 = MyDataClass(field1="value1")
    obj2 = MyDataClass(field1="value2")
    
    # Test mutable default isolation
    obj1.mutable_field.append("item1")
    assert len(obj2.mutable_field) == 0  # Catches shared default mutation
    
    # Test immutability if frozen=True
    if hasattr(MyDataClass, '__dataclass_fields__'):
        frozen = getattr(MyDataClass.__dataclass_params__, 'frozen', False)
        if frozen:
            with pytest.raises(FrozenInstanceError):
                obj1.field1 = "new_value"
    
    # Test post-init behavior
    if hasattr(obj1, 'computed_field'):
        assert obj1.computed_field is not None  # Catches __post_init__ removal
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run Mutation Testing with Modern Python Features
  run: |
    sloptest generate tests/ --enable-mutation-testing
    sloptest analyze --minimum-mutation-score 85
```

### Coverage Integration
The modern mutators integrate with existing coverage analysis to ensure comprehensive testing of:
- Type annotation edge cases
- Async/await error paths  
- Dataclass configuration scenarios

This provides a complete picture of test quality for modern Python codebases.

## Troubleshooting

### Common Issues

1. **High Async Mutation Survival**: Often indicates missing tests for concurrent behavior
2. **Type Hint Mutation Survival**: Usually means tests don't validate type constraints
3. **Dataclass Mutation Survival**: Typically missing tests for immutability or shared state

### Performance Considerations

- Modern mutators generate more mutations than basic operators
- Async tests may take longer to execute
- Consider using `max_mutants_per_file` to limit scope for large files

### Debugging Failed Mutations

Use the detailed mutation reports to identify which specific mutations are surviving:

```bash
sloptest analyze --verbose --mutation-details
```

This shows exactly which modern Python patterns need better test coverage.

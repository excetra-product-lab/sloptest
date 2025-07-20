"""Cost management utilities for LLM API usage."""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CostManager:
    """Manages LLM API costs through various optimization strategies."""
    
    def __init__(self, config):
        self.config = config
        self.token_usage_log = []
        
        # Cost management settings
        self.max_file_size_kb = config.get('cost_management.max_file_size_kb', 50)
        self.max_context_size_chars = config.get('cost_management.max_context_size_chars', 100000)
        self.max_files_per_request = config.get('cost_management.max_files_per_request', 15)
        self.use_cheaper_model_threshold_kb = config.get('cost_management.use_cheaper_model_threshold_kb', 10)
        self.enable_content_compression = config.get('cost_management.enable_content_compression', True)
        self.skip_trivial_files = config.get('cost_management.skip_trivial_files', True)
        self.token_usage_logging = config.get('cost_management.token_usage_logging', True)
        
        # Model pricing (tokens per dollar - approximate)
        self.model_costs = {
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},  # per 1K tokens
            'claude-3-5-haiku-20241022': {'input': 0.00025, 'output': 0.00125},
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'claude-3-7-sonnet-20250219': {'input': 0.003, 'output': 0.015},
            'claude-sonnet-4-20250514': {'input': 0.003, 'output': 0.015},
            'claude-opus-4-20250514': {'input': 0.015, 'output': 0.075}
        }
    
    def should_skip_file(self, file_path: str) -> Tuple[bool, str]:
        """Determine if file should be skipped for cost reasons."""
        try:
            file_size_kb = os.path.getsize(file_path) / 1024
            
            # Skip large files
            if file_size_kb > self.max_file_size_kb:
                return True, f"File too large ({file_size_kb:.1f}KB > {self.max_file_size_kb}KB)"
            
            # Skip trivial files if enabled
            if self.skip_trivial_files:
                if self._is_trivial_file(file_path):
                    return True, "File has minimal testable content"
            
            return False, ""
            
        except Exception as e:
            logger.warning(f"Error checking file {file_path}: {e}")
            return False, ""
    
    def _is_trivial_file(self, file_path: str) -> bool:
        """Check if file has minimal testable content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count functions and classes
            function_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
            class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
            
            # Skip files with very few testable elements
            return (function_count + class_count) < 5
            
        except Exception:
            return False
    
    def optimize_batch_size(self, files: List[str]) -> int:
        """Determine optimal batch size based on file sizes."""
        if not files:
            return self.max_files_per_request
        
        # Calculate average file size
        total_size = 0
        valid_files = 0
        
        for file_path in files:
            try:
                size_kb = os.path.getsize(file_path) / 1024
                total_size += size_kb
                valid_files += 1
            except Exception:
                continue
        
        if valid_files == 0:
            return self.max_files_per_request
        
        avg_size_kb = total_size / valid_files
        
        # Adjust batch size based on average file size
        if avg_size_kb > 20:
            return min(5, self.max_files_per_request)  # Large files: smaller batches
        elif avg_size_kb > 10:
            return min(10, self.max_files_per_request)  # Medium files: medium batches
        else:
            return self.max_files_per_request  # Small files: larger batches
    
    def suggest_model_for_files(self, files: List[str]) -> str:
        """Suggest the most cost-effective model for given files."""
        if not files:
            return "claude-3-5-sonnet-20241022"  # Default
        
        # Calculate total size
        total_size_kb = 0
        for file_path in files:
            try:
                total_size_kb += os.path.getsize(file_path) / 1024
            except Exception:
                continue
        
        avg_size_kb = total_size_kb / len(files) if files else 0
        
        # Use cheaper model for smaller, simpler files
        if avg_size_kb < self.use_cheaper_model_threshold_kb:
            return "claude-3-haiku-20240307"  # Cheaper for simple files
        
        # Check complexity indicators
        high_complexity = self._has_high_complexity(files)
        
        if high_complexity:
            return "claude-3-7-sonnet-20250219"  # Better for complex logic
        else:
            return "claude-3-5-sonnet-20241022"  # Good balance
    
    def _has_high_complexity(self, files: List[str]) -> bool:
        """Check if files contain complex patterns requiring better models."""
        complexity_indicators = [
            r'@\w+\(',  # Decorators
            r'async\s+def',  # Async functions
            r'yield\s+',  # Generators
            r'metaclass\s*=',  # Metaclasses
            r'__\w+__\s*=',  # Dunder methods
            r'typing\.',  # Complex typing
        ]
        
        for file_path in files[:3]:  # Check first 3 files only for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                complexity_score = sum(
                    len(re.findall(pattern, content, re.MULTILINE))
                    for pattern in complexity_indicators
                )
                
                if complexity_score > 5:  # Arbitrary threshold
                    return True
                    
            except Exception:
                continue
        
        return False
    
    def compress_content(self, content: str) -> str:
        """Remove unnecessary content to reduce token usage."""
        if not self.enable_content_compression:
            return content
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove single-line comments (but keep docstrings)
        content = re.sub(r'^\s*#(?!.*"""|\'\'\').*$', '', content, flags=re.MULTILINE)
        
        # Remove empty lines after imports
        content = re.sub(r'(import\s+.*?)\n\n+', r'\1\n', content)
        
        return content.strip()
    
    def log_token_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Log token usage for cost tracking."""
        if not self.token_usage_logging:
            return
        
        cost_data = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost': self._calculate_cost(model, input_tokens, output_tokens)
        }
        
        self.token_usage_log.append(cost_data)
        
        # Log to file for persistence
        self._save_usage_log()
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for token usage."""
        if model not in self.model_costs:
            return 0.0
        
        costs = self.model_costs[model]
        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        
        return input_cost + output_cost
    
    def _save_usage_log(self):
        """Save token usage log to file."""
        try:
            log_file = Path('.testgen_usage.json')
            
            # Load existing log
            existing_log = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing_log = json.load(f)
            
            # Append new entries
            existing_log.extend(self.token_usage_log)
            
            # Keep only last 100 entries
            existing_log = existing_log[-100:]
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(existing_log, f, indent=2)
            
            # Clear in-memory log
            self.token_usage_log.clear()
            
        except Exception as e:
            logger.warning(f"Failed to save usage log: {e}")
    
    def get_usage_summary(self, days: int = 7) -> Dict:
        """Get cost usage summary for specified days."""
        try:
            log_file = Path('.testgen_usage.json')
            if not log_file.exists():
                return {'total_cost': 0, 'requests': 0, 'total_tokens': 0}
            
            with open(log_file, 'r') as f:
                usage_log = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_usage = [
                entry for entry in usage_log
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            
            total_cost = sum(entry.get('estimated_cost', 0) for entry in recent_usage)
            total_requests = len(recent_usage)
            total_tokens = sum(
                entry.get('input_tokens', 0) + entry.get('output_tokens', 0)
                for entry in recent_usage
            )
            
            return {
                'total_cost': round(total_cost, 4),
                'requests': total_requests,
                'total_tokens': total_tokens,
                'average_cost_per_request': round(total_cost / max(total_requests, 1), 4)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get usage summary: {e}")
            return {'total_cost': 0, 'requests': 0, 'total_tokens': 0}
    
    def get_optimization_suggestions(self, files: List[str]) -> List[str]:
        """Get cost optimization suggestions for the current batch."""
        suggestions = []
        
        # Check file sizes
        large_files = []
        for file_path in files:
            try:
                size_kb = os.path.getsize(file_path) / 1024
                if size_kb > self.max_file_size_kb * 0.8:  # 80% of limit
                    large_files.append((file_path, size_kb))
            except Exception:
                continue
        
        if large_files:
            suggestions.append(
                f"Consider splitting large files: {', '.join(f'{Path(f).name} ({s:.1f}KB)' for f, s in large_files[:3])}"
            )
        
        # Check batch size
        optimal_batch = self.optimize_batch_size(files)
        if len(files) > optimal_batch:
            suggestions.append(
                f"Consider reducing batch size to {optimal_batch} for better cost efficiency"
            )
        
        # Model suggestion
        suggested_model = self.suggest_model_for_files(files)
        suggestions.append(f"Recommended model for this batch: {suggested_model}")
        
        return suggestions 
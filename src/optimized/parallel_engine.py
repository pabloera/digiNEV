"""
Minimal parallel engine implementation for TDD Phase 3.
Implements basic parallel processing without psutil dependency.
"""

import concurrent.futures
from typing import Any, Callable, List, Optional


class ParallelEngine:
    """
    Minimal parallel processing engine to pass TDD tests.
    
    Implements basic parallel processing without external dependencies,
    following TDD principles.
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize parallel engine."""
        self.max_workers = max_workers
        self.pool_size = max_workers
    
    def process_parallel(self, func: Callable, data: List[Any], max_workers: Optional[int] = None) -> List[Any]:
        """Process data in parallel using ThreadPoolExecutor."""
        workers = max_workers or self.max_workers
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                futures = [executor.submit(func, item) for item in data]
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # For TDD - handle errors gracefully
                        results.append({'error': str(e), 'success': False})
                
                return results
                
        except Exception as e:
            # Fallback to sequential processing
            results = []
            for item in data:
                try:
                    result = func(item)
                    results.append(result)
                except Exception as error:
                    results.append({'error': str(error), 'success': False})
            return results
    
    def map(self, func: Callable, data: List[Any]) -> List[Any]:
        """Map function over data (alias for process_parallel)."""
        return self.process_parallel(func, data)


# Global instance for TDD
_global_parallel_engine = None


def get_global_parallel_engine() -> Optional[ParallelEngine]:
    """Get global parallel engine instance."""
    global _global_parallel_engine
    
    if _global_parallel_engine is None:
        try:
            _global_parallel_engine = ParallelEngine()
        except Exception:
            # Return None if can't initialize
            return None
    
    return _global_parallel_engine

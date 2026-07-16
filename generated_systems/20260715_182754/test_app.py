import pytest
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import Service

class TestService:
    def test_initial_value(self):
        obj = Service()
        assert obj.get() == 0
    
    def test_increment(self):
        obj = Service()
        obj.increment()
        assert obj.get() == 1
    
    def test_multiple_increments(self):
        obj = Service()
        for i in range(10):
            obj.increment()
        assert obj.get() == 10
    
    def test_concurrent_increments(self):
        obj = Service()
        threads = []
        
        def worker():
            for _ in range(100):
                obj.increment()
        
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert obj.get() == 1000

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
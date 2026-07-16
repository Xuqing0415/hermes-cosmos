import pytest
import threading

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
        return self.value
    
    def get(self):
        with self.lock:
            return self.value

class TestCounter:
    def test_initial_value(self):
        obj = Counter()
        assert obj.get() == 0
    
    def test_increment(self):
        obj = Counter()
        obj.increment()
        assert obj.get() == 1
    
    def test_multiple_increments(self):
        obj = Counter()
        for i in range(10):
            obj.increment()
        assert obj.get() == 10
    
    def test_concurrent_increments(self):
        obj = Counter()
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
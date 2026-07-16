from flask import Flask, jsonify, request
import threading
import json
import time
import uuid

app = Flask(__name__)

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
        self._state_history = []
    
    def get_state(self):
        return {"value": self.value}
    
    def record_transition(self, operation, pre_state, post_state):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "operation": operation,
            "pre_state": pre_state,
            "post_state": post_state
        }
        self._state_history.append(entry)
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-500:]
        return entry

    def increment(self):
        with self.lock:
            self.value += 1
        return self.value

    def get(self):
        with self.lock:
            return self.value

counter = Counter()

@app.before_request
def before_request():
    request._start_time = time.time()
    request._pre_state = counter.get_state().copy()

@app.after_request
def after_request(response):
    try:
        if hasattr(request, '_pre_state'):
            post_state = counter.get_state().copy()
            operation = request.endpoint or request.path
            counter.record_transition(operation, request._pre_state, post_state)
    except Exception:
        pass
    return response

@app.route('/inc', methods=['POST'])
def increment():
    result = counter.increment()
    return jsonify({'value': result})

@app.route('/get', methods=['GET'])
def get():
    result = counter.get()
    return jsonify({'value': result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "value": counter.get_state()["value"]})

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(counter.get_state())

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({
        "history": counter._state_history,
        "count": len(counter._state_history)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
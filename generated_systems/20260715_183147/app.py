from flask import Flask, jsonify
import threading

app = Flask(__name__)

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

counter = Counter()


@app.route('/inc', methods=['POST'])
def increment():
    
    result = counter.increment()
    return jsonify({'value': result})
    

@app.route('/get', methods=['GET'])
def get():
    
    result = counter.get()
    return jsonify({'value': result})
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
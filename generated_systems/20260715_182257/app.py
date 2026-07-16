from flask import Flask, jsonify
import threading

app = Flask(__name__)

class Service:
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

service = Service()


@app.route('/read', methods=['GET'])
def read():
    
    return jsonify({'error': 'Not implemented'}), 501
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
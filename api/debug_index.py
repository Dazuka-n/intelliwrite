from flask import Flask

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return f"Caught path: {path}"

if __name__ == '__main__':
    app.run(debug=True)

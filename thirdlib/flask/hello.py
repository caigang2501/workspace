from flask import Flask

app = Flask(__name__)

@app.route('/index/pa')
def index():
    return 'Index Page111'

@app.route("/user/<name>")
def hello_world(name):
    print('asdffa')
    return f"Hello, cnmq{name}"

# app.run(host='0.0.0.0',debug=True)
app.run(host='0.0.0.0')
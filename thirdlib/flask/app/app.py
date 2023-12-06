from flask import Flask,request,jsonify
import logging
from funcs import funcs
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(filename='app.log', level=logging.DEBUG)

@app.route("/test_get", methods=['GET'])
def test_get():
    result = funcs.main([1,2,3])
    app.logger.info(__name__)
    return jsonify(__name__)

@app.route("/test_post", methods=['POST'])
def test_post():
    data = request.get_json()
    result = funcs.main(data)
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0')
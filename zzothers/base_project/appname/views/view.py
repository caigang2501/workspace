import sys,os
sys.path.append(os.getcwd())

from flask import request, jsonify,Blueprint
import main
import logging

bp = Blueprint("main", __name__)

@bp.route("/test", methods=['POST'])
def fun1():
    data = request.get_json()['points']
    result = main.mst_edge(data)
    return jsonify(result)

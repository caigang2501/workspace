import sys,os

from flask import request, jsonify,Blueprint,send_file
import main
import logging

bp = Blueprint("main", __name__)

# "http://127.0.0.1:5000/output_logs?plan_id=12341241234"
@bp.route("/output_logs", methods=['GET'])
def output_logs():
    pname = request.args.get('plan_id')
    if pname=='latest':
        with open('example/result/latest_plan_id.txt', 'r') as file:
            pname = file.read()
    file_path = os.path.join(os.getcwd()+'/example/output',f'{pname}.xlsx')
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return 'no such file'

@bp.route("/test", methods=['GET'])
def test():
    a = main.test1()
    return jsonify(a)




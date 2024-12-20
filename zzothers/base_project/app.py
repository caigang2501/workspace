import sys,os,socket
from flask import Flask
from appname.views import gp_view

def create_app():
    app = Flask(__name__)
    # 注册 Blueprint
    app.register_blueprint(gp_view.bp)

    return app

if __name__ == "__main__":
    if 'WINDOWS' in socket.gethostname():
        create_app().run(debug=True) # threaded=True processes=True
    else:
        create_app().run()





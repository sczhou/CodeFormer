import argparse

from flask import Flask, Blueprint, jsonify

from handlers import restore_face_route_v1
from utils.logs import log

# create a parser object
parser = argparse.ArgumentParser(description="A Flask app to frontend Codeformer")

# add arguments
parser.add_argument('--port', type=int, default=5000, help='The port to run the server on')
parser.add_argument('--prefix', type=str, default='codeformer', help='The route prefix for server to use')
parser.add_argument('--host', type=str, default='127.0.0.1', help='The host')

args = parser.parse_args()

# create a flask app with port, server as command line arguments
app = Flask(__name__)
prefix_route = Blueprint(args.prefix, __name__, url_prefix=f'/{args.prefix}')
prefix_route.register_blueprint(restore_face_route_v1)

app.register_blueprint(prefix_route)


@app.errorhandler(404)
def page_not_found(error):
    log.error(error)
    data = {"data": {"message": "Invalid route"}, "err": {}}
    return jsonify(data), 404


@app.errorhandler(405)
def method_not_allowed(error):
    log.error(error)
    data = {"data": {"message": "Method not allowed"}, "err": {}}
    return jsonify(data), 405


# run the app
if __name__ == '__main__':
    app.run(port=args.port, host=args.host)

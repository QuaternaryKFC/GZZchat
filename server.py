import os

from flask import Flask, send_from_directory

# start flask server and set public folder and url
from utils.evaluate import Predictor
predictor = Predictor('configs.server_config')
app = Flask(__name__, static_folder='gzzchat/build', static_url_path='')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# route for showing details of movie
@app.route('/reply/<sentence>', methods=['get'])
def reply(sentence):
    return predictor.predict(sentence)


if __name__ == '__main__':
    app.run()

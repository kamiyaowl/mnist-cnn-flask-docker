# GPUは使わない
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import tensorflow as tf
from tensorflow.python import keras

from flask import Flask, jsonify, abort, make_response, request, send_from_directory
import numpy as np

graph = tf.get_default_graph()
model = None

app = Flask(__name__)

# 疎通確認
@app.route("/info")
def index():
    return make_response(jsonify({
        "name": "mnist-cnn server",
        "time": time.ctime(),
    }))

# 28*28の画像をPOSTで送ると、0~9の推論結果を返してくれる
@app.route("/predict", methods=['POST'])
def mnist():
    data = request.json
    if data == None:
        return abort(400)
    src = data["src"]
    if (src == None) | (not isinstance(src, list)):
        return abort(400)
    src = np.array(src)
    # 正規化する
    src = src.astype('float32') / 255.0
    src = src.reshape(-1,28,28,1)
    # 推論する
    with graph.as_default():
        start = time.time()
        dst = model.predict(src)
        elapsed = time.time() - start
        return make_response(jsonify({ 
            "predict" : dst.tolist(),
            "elapsed" : elapsed,
        }))

# 静的ファイル公開    
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def send_file(path):
    return send_from_directory("dist", path)


if __name__ == '__main__':
    model = keras.models.load_model("./model.h5")
    app.run(host="0.0.0.0", port=3000, debug=True)
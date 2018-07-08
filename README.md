# mnist-cnn-flask-docker

## Overview

tensorflow/kerasで作成した手書き文字認識のモデルを、Webアプリ経由で利用するサンプルです。

フロントはvue.js、バックエンドはpython+flaskで動作しています。

## Demo

### Docker Compose

`$ docker-compose up`

### Docker

```
$ docker build -t mnist-cnn-flask-docker .
$ docker run -p 3000:3000 -it mnist-cnn-flask-docker
```

### ローカルのpythonで実行

`$ python mnist-server.py`
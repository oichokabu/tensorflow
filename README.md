# Tensorflow使い方
```s
$ pip3 install --upgrade pip
$ pip3 install tensorflow
$ pip3 install tf-nightly
$ docker pull tensorflow/tensorflow:latest  # あらかじめdockerクライアントアプリを起動しておく
$ docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://85a1eebba5ed:8888/?token=45347f34f9430aac93defc267b7c593ba6ca7edae1063cb8
     or http://127.0.0.1:8888/?token=45347f34f9430aac93defc267b7c593ba6ca7edae1063cb8
```

- https://tail-island.github.io/programming/2016/04/25/primer-of-tensorflow-1.html
   - tensorBoard.py
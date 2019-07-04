FROM tensorflow/tensorflow:1.13.1-py3
COPY inference.py private_detector.frozen.pb /workdir/
WORKDIR /workdir
ENTRYPOINT ["python3", "inference.py"]

FROM tensorflow/tensorflow:1.15.0-py3
USER root
RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --fix-missing && \
    apt-get install -y git

RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN pip3 install Keras keras-ocr
RUN pip3 install Pillow six Flask flask_bootstrap numpy requests Werkzeug tensorflow-serving-api==1.15.0
RUN apt-get install -y libsm6 libxrender1 vim
#CMD ["echo", "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal"]
#RUN sh -c 'tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -'
#RUN apt-get install -y  tensorflow-model-server
#EXPOSE 8500
#EXPOSE 8501
RUN mkdir -p tfserving-scripts/License_Detection_Model/
COPY run_server.sh tfserving-scripts/
COPY License_Detection_Model/ tfserving-scripts/License_Detection_Model/
RUN mkdir flaskapp
COPY flaskapp/ flaskapp/
RUN echo deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal >> /etc/apt/sources.list.d/tensorflow-serving.list 
RUN apt-get install -y ca-certificates wget
#RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN wget -O- https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update
RUN apt-get install tensorflow-model-server
RUN git clone https://github.com/fizyr/keras-retinanet
RUN cd keras-retinanet && pip3 install .
#ENTRYPOINT ["/tfserving-scripts/run_server.sh]

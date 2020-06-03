from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, os.path
import re
import sys
import tarfile
import copy
import sys
import time, random
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
i#import textwrap
import numpy as np
from six.moves import urllib
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import cv2
# !flask/bin/python
from flask import Flask, jsonify, flash, Response
from flask import make_response
from flask import request, render_template
from flask_bootstrap import Bootstrap
from flask import redirect, url_for
from flask import send_from_directory, send_file
from flask import Flask, make_response
from werkzeug.utils import secure_filename
from subprocess import call
from flask import jsonify
# from sightengine.client import SightengineClient

from flask_util import ListConverter
import io
import zlib
import requests
from io import BytesIO

from PIL import Image

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color 

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# from text_recognition import TextRecognition
# from imutils.object_detection import non_max_suppression
# import pytesseract
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

print('imported Libs ...!')

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = '/home/dl-highgpu/detection_projects/licenseplate_india/flask_save_images'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
bootstrap = Bootstrap(app)

app.url_map.converters['list'] = ListConverter

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
                    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model_path = 'snapshots/resnet50_csv_20.h5'
classes_file = 'classes.csv'

@app.route('/', methods=['GET', 'POST'])
def ocr_upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
#         files = request.files.getlist('file')
#         print(files)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
#         if len(files) == 0:
            flash('No selected file')
            return redirect(request.url)

        filename = file.filename
        if file and allowed_file(filename):
            filename = secure_filename(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            savepath = predict_detection_image(filename)
            fname = os.path.basename(savepath)
        return redirect(url_for('uploadDetectedImage',
                                        filename=fname))

    return render_template("input.html")

#         filenames = list()
#         for file in files:
#             filename = file.filename
#             if file and allowed_file(filename):

#                 filename = secure_filename(filename)
#                 filenames.append(filename)
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# #         print(len(filenames), filenames[0])
#             # fix_orientation(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# #         return redirect(url_for('predict_detection_image',
# #                                         filenames=filenames))
#         predict_detection_image
#     return render_template("input.html")


def run(image_path, model, host='localhost', port=8500, signature_name='serving_default'):
    
    options=[
                      ('grpc.max_send_message_length', 50 * 1024 * 1024),
                                ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                                      ]
    
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port), options = options)
    
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    image = read_image_bgr(image_path)
    
    image = preprocess_image(image.copy())
    image, scale = resize_image(image)
    
    data = np.array(image).astype(tf.keras.backend.floatx())
    
    start = time.time()
    
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    print('Shape of data going in to inputs',data.shape)
    request.inputs['input_image'].CopyFrom(make_tensor_proto(data, shape=[1, data.shape[0], data.shape[1], 3]))

    result = stub.Predict(request, 10.0)
    
    end = time.time()
    time_diff = end - start
    
    print('time elapased: {}'.format(time_diff))
    print(type(result))
#     print('classes',result.outputs['classes'])
    
    bboxes_proto = result.outputs["filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0"]
#     print('bboxes proto',bboxes_proto)
    bboxes_proto_shape = tf.TensorShape(bboxes_proto.tensor_shape)
#     print('bboxes_proto_shape',bboxes_proto.tensor_shape)
    bboxes = tf.constant(bboxes_proto.float_val, shape=bboxes_proto_shape)/scale
    
    print(bboxes.numpy().shape)
    
    confidences_proto = result.outputs["filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0"]
    confidences_proto_shape = tf.TensorShape(confidences_proto.tensor_shape)
    confidences = tf.constant(confidences_proto.float_val, shape=confidences_proto_shape)

    labels_proto = result.outputs["filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0"]
    labels_shape = tf.TensorShape(labels_proto.tensor_shape)
    labels = tf.constant(labels_proto.int_val, shape=labels_shape)

    return bboxes.numpy(), confidences.numpy(), labels.numpy()

def get_cropped_images_nms(bboxes, scores, labels, imagepath,threshold):
    
    img = Image.open(imagepath)
    
    originalImage = np.asarray(img)
    
    bboxes_res = []
    scores_res = []
    labels_res = []
    for index, value in enumerate(scores):
        
        if labels[index] < 0 or value < threshold:
            break
        if labels[index] == 0:
            bboxes_res.append(bboxes[index])
            scores_res.append(value)
            labels_res.append(labels[index])
        else:
            print(labels[index], 'in else')
            
    print(len(bboxes_res), len(scores_res))
    print(bboxes_res)
    print(scores_res)
    
    return bboxes_res, scores_res, labels_res


def draw_detections(image, boxes, scores, labels):
    
    for index, value in enumerate(scores):
        color = label_color(labels[index])
        b = boxes[index].astype(int)
        draw_box(image, b, color=color)
        caption = "{} {:.3f}".format('License Plate', value)
        draw_caption(image, b, caption)
    
      

def saveDetectedImage(imagePath,outpath ,boxes, scores, labels, texts):
    
    image = read_image_bgr(imagePath)
    
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    draw_detections(draw, boxes, scores, labels)
    font = cv2.FONT_HERSHEY_SIMPLEX 
#     print(boxes[0][2], len(boxes),"Boxes")
#     org = (20,20)
    for index, box in enumerate(boxes):
        x = int(box[2])
        y = int(box[3] + 5.0)
        org = (x, y) 
        fontScale = 0.75
        color = (0, 0, 255) 
        thickness = 2
        draw = cv2.putText(draw,texts[index], org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
#     print(type(draw))
    cv2.imwrite(outpath, draw)
    return outpath
#     return redirect(url_for('uploadDetectedImage',
#                                         filename=outpath))
    
    

@app.route('/detected_image/<filename>')

def uploadDetectedImage(filename):
    
    fname = os.path.basename(filename)
    
    #html not recognizing path of image ????
    
#     return render_template('final_results.html', user_image=fname)
#     img_url = 'http://127.0.0.1:5000/flask_save_images/' + fname
#     print(img_url, "fname")
#     return render_template('final_results.html', filename=img_url)

    #working
    return send_from_directory(app.config['IMAGE_FOLDER'],
                               fname)



#     img = Image.fromarray(draw.astype('uint8'))
#     # create file-object in memory
#     file_object = io.BytesIO()

#     # write PNG in file-object
#     img.save(file_object, 'PNG')

#     # move to beginning of file so `send_file()` it will read from start    
#     file_object.seek(0)
    
#     return send_file(file_object, mimetype='image/PNG')

    
# @app.route('/image_detection/<list:filenames>')
# @app.route('/image_ocr/<filename>')
def predict_detection_image(filename):
    
#     for filename in filenames:
        
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    save_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
    bboxes, scores, labels = run(img_path, model='License_Detection_Model', host = '13.82.136.16')
    final_bboxes, finalscores, finallabels = get_cropped_images_nms(bboxes=bboxes[0] , scores= scores[0], labels= labels[0], 
                                                                    imagepath= img_path,threshold=0.7)
    texts = []
    for box in final_bboxes:
        img = Image.open(img_path)
        im = img.crop(box)  
        im.save('plt.jpg')
        text = predict_text(np.asarray(im.convert('RGB')) )
        texts.append(text)
#     print(save_path)
    saveDetectedImage(img_path,save_path, final_bboxes, finalscores, finallabels, texts)

    return save_path
        


# #skew correction
# def deskew(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#      if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# #template matching
# def match_template(image, template):
#     return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)     
    

def predict_text(image_crop):
    
    text = ""
    images = ['plt.jpg']
    prediction_groups = pipeline.recognize(images)
    for groups in prediction_groups:
        for group in groups:
            text = text + group[0]
    print(text.upper())
    return text.upper()
    
    


#     gray = get_grayscale(image_crop)
    
#     thresh = thresholding(gray)
# #     opening = opening(gray)
#     can = canny(gray)
# #     cv2.imwrite('canny.jpg',can)
#     custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6'
# #     custom_config = r'--oem 3 --psm 6'
#     print(type(canny))
#     res = pytesseract.image_to_string(canny, config=custom_config)
#     print(res)
        
    
# def initRecogModel():
#     ''' Initializes recognition Model'''
#     print('in initRecogModel ..')
#     recognition_pb = '/home/dl-highgpu/detection_projects/licenseplate_india/recognition/zhang_checkpoints/checkpoint/text_recognition.pb'
#     with tf.device('/gpu:0'):
#         tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
#                                        allow_soft_placement=True)
#         recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
#     np_load_old = np.load

#     # modify the default parameters of np.load
#     np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#     label_dict = np.load('./reverse_label_dict_with_rects.npy')[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
#     return recognition_model, label_dict

# ocr_recognition_model ,  ocr_label_dict = initRecogModel()

# def detection_crop_array(imgArr, recognition_model,label_dict):
    
#     height, width = imgArr.shape[:2]
#     test_size = 299
#     scale = test_size / width
#     resized_image = cv2.resize(imgArr, (0, 0), fx=scale, fy=scale)
    
#     print(resized_image.shape)
#     top_bordersize = (test_size - resized_image.shape[0]) // 2
#     bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
#     image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
#                                       right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     cv2.imwrite('plate.jpg',image_padded)
#     image_padded = np.float32(image_padded) / 255.
#     image_padded = np.expand_dims(image_padded, 0)
#     print(image_padded.shape)
    
#     results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
#     return results, probs


# def convert_array_chars_tostr(chararray):
#     res1 = ''
#     if len(chararray) == 0:
#         return res1
#     for i in range(len(chararray)):
#         res1 = res1 + chararray[i]
#     return res1

# def predict_text(image_crop):
    
#     print('in predict_text')
    
#     results, probs = detection_crop_array(image_crop, ocr_recognition_model, ocr_label_dict)
#     text = convert_array_chars_tostr(results)
#     print(text)
    



if __name__ == '__main__':
        # os.environ["TF_ENABLE_CONTROL_FLOW_V2"] = "0"
    app.run(host='0.0.0.0', debug=True)


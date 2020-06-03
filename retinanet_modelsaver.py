import tensorflow as tf
import os, re
from glob import glob

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
#from keras_retinanet.utils.gpu import setup_gpu
#from keras_retinanet.utils.keras_version import check_keras_version
#from keras_retinanet.utils.tf_version import check_tf_version


def exporter(model, export_path_base):
    versions = []
    for i in glob(os.path.join(export_path_base,'*')):
        if os.isdir(i) and re.match(r'^[0-9]+$',i):
            versions.append(i)
    if versions == []:
        version = 0
    else:
        version = len(versions)

    export_path = os.path.join(export_path_base, str(version))
    os.makedirs(export_path)

    #disable dropout and other train only ops
    tf.keras.backend.set_learning_phase(0)

    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name:t for t in model.outputs})

    print('exported model to {}'.format(export_path))


def main():

#     config_file = '/home/segmind/Desktop/PRATIK/TF-SERVING/snapshots/config.ini'
    model_in = 'resnet50_csv_50.h5'
    model_out_base = '/tmp/keras_retinanet'
    backbone = 'resnet50'

    # optionally load config parameters
    #anchor_parameters = None
    #if args.config:
#     args_config = read_config_file(config_file)
#     anchor_parameters = parse_anchor_parameters(args_config)

    # load the model
    model = models.load_model(model_in, backbone_name=backbone)

    # check if this is indeed a training model
    models.check_training_model(model)

    # convert the model
    print('Building inference model ..')
    model = models.convert_model(model, nms=True, class_specific_filter=True)

    print('exporting with tf-serving ..')
    exporter(model, export_path_base=model_out_base)

    # save model
    #model.save(args.model_out)


if __name__ == '__main__':
    main()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from tensorflow import keras

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image



from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90



if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	    file_name = os.path.basename(file.name)
	    if 'frozen_inference_graph.pb' in file_name:
	        tar_file.extract(file, os.getcwd())
	    print ('Download complete')
else:
	    print ('Model already exists')



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(os.path.join('/home/saboor/Desktop/Object-detection/models/research/object_detection/data','mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
categories_index = label_map_util.create_category_index(categories)




import cv2
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      ret,image_np = cap.read()
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          categories_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      cv2.imshow('image', cv2.resize(image_np, (1280, 960)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break


keras_file = "linera.h5"

keras.models.save_model(model,keras)

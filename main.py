import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


from  collections import defaultdict
from io import StringIO
import tkinter

import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print ("testing", gui)
        matplotlib.use(gui, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print ("Using:",matplotlib.get_backend())
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

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


def load_img_into_numpy_array(image):
  (im_width,im_height) = image.size
  return  np.array(image.getdata()).reshape((im_width,im_height,3)).astype(np.uint8)

PATH_TO_IMG_DIR = '/home/saboor/Desktop/Object-detection/test_images/'
TEST_IMG_PATH = [os.path.join(PATH_TO_IMG_DIR,'image{}.jpeg'.format(i)) for i in range(1,3)]

IMAGE_SIZE = (12,8)

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMG_PATH:
      image = Image.open(image_path)
      image_np = load_img_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np,axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      (boxes,scores,classes,num_detections) = sess.run([boxes,scores,classes,num_detections],
                                                       feed_dict={image_tensor: image_np_expanded})

      vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),
                                                         np.squeeze(classes).astype(np.int32),
                                                         np.squeeze(scores),
                                                         categories_index,
                                                         use_normalized_coordinates=True,
                                                         line_thickness=8)

      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()








import os
import cv2
import itertools
import numpy as np
import json
import tensorflow as tf
from multiprocess import Queue, Pool
import sys
import tensorflow as tf
from threading import Thread
from datetime import datetime
from collections import defaultdict

######## Global Variables ########

detection_graph = None
sys.path.append("..")
_score_thresh = 0.27
_hand_thresh_ = None
PATH_TO_CKPT = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'hand_label_map.pbtxt'
NUM_CLASSES = 1
label_map = None
categories = None
category_index = None


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def get_rect_points(boxes, im_width, im_height):

    (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                          boxes[0][0] * im_height, boxes[0][2] * im_height)
    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    return left, top, right, bottom

# COLOR THRESHOLDS FOR HANDS

def load_thresh(filename='hand_thresh_final'):
    res = None
    with open(filename + '.json', mode='r') as fp:
        res = json.load(fp)
    for k in res.keys():
        for i in range(len(res[k])): 
            if len(res[k][i]) == 0:
                res[k][i] = [0,256]
    return res



def save_thresh(_hdthresh_, filename='hand_thresh'):
    with open(filename + '.json', mode='w') as fp:
        json.dump(_hdthresh_, fp, indent=2)

# THRESHOLDING UTILITIES

def cvt_color_space(img:np.ndarray, source, dest):
    """ Converts image from one colorspace to another. Allowed colorspaces are: HSV, 
    YCRCB, RGB, BGR, GRAY(scale)."""
    
    if source in ['ycrcb', 'ycbcr', 'YCrCb', 'YCRCB', 'YCBCR']:
        source = 'ycr_cb'
    if dest in ['ycrcb', 'ycbcr', 'YCrCb', 'YCRCB', 'YCBCR']:
        dest = 'ycr_cb'

    source = source.upper()
    dest = dest.upper()
    modes = ["BGR", "RGB", "GRAY", "HSV", "YCR_CB"]
    assert source in modes and dest in modes
    if source == dest:
        # print("(warning): transforming into same color space won't cause any change")
        return img.copy()
    else:
        try:
            s = 'cv2.cvtColor(img, cv2.COLOR_{0}2{1})'.format(source, dest)
            return eval(s)
        except TypeError as e:
            raise ValueError("""Caught exception while evaluating command string in color conversion, with text:
                                    '{}'
            This is probably due to wrong usage of image as parameter passed to this function. Check again to have
            used 'Egohands.access([...]).VALUES()' to correctly retrieve image and mask (they are returned as dict).
            """.format(e))


def thresh_mask(img:np.ndarray, t1=[0,256], t2=[0,256], t3=[0,256]):
    return np.all(
            np.stack([
            (t1[0] <= img[:,:,0]) & (img[:,:,0] < t1[1]),
            (t2[0] <= img[:,:,1]) & (img[:,:,1] < t2[1]),
            (t3[0] <= img[:,:,2]) & (img[:,:,2] < t3[1])
            ], axis=2),
        axis=2)


def color_thresholding(img:np.ndarray, t1=[0,256], t2=[0,256], t3=[0,256], source="bgr", dest="rgb"):
    """Applies specified thresholds to the 3 channels of the image. The 'source' and 'dest' parameters
    can be used when an image is from a colorspace but we want to compute the thresholding in another."""
    _color_black_ = {
        'rgb': np.array([0,0,0], dtype=np.uint8),
        'bgr': np.array([0,0,0], dtype=np.uint8),
        'hsv': np.array([0,0,0], dtype=np.uint8),
        'ycr_cb': np.array([16, 128, 128], dtype=np.uint8)
    }

    assert len(t1) == len(t2) == len(t3) == 2
    res = cvt_color_space(img, source, dest)
    bmask = thresh_mask(img, t1, t2, t3)
    res[np.logical_not(bmask)] = _color_black_[dest]
    return res
    

def colorspace_mask(img:np.ndarray, colorspace='bgr'):
    global _hand_thresh_
    return thresh_mask(img, *_hand_thresh_[colorspace])

def perform_hand_thresholding(img:np.ndarray, mode="rgb", dest=None):
    """
        Performs color thresholding to recognize hands. The 'dest' parameter can be used when an image belongs
        to a colorspace but the thresholding is wanted to be computed in anothr one.
    """
    # NOTE: this can be problematic, but have not found yet a workaround
    if dest is None:
        dest = mode
    return color_thresholding(
        img, 
        t1=_hand_thresh_[mode][0], 
        t2=_hand_thresh_[mode][1], 
        t3=_hand_thresh_[mode][2], 
        source=mode, 
        dest=dest
    )

# Most useful function!!
def threshold_rect(frame,colorspace,x,y,w,h,from_space='bgr'):
    """Performs thresholding given a specific colorspace, inside a provided rectangle (the point (x,y) is the upper left, while (x+w, y+h) is the lower right. 
    Returns the mask over the full image.
    
    **NOTE**: this is thought to work RT along with OpenCV, therefore performs automatically conversion from BGR."""
    thresholded = np.zeros_like(frame)
    mask = colorspace_mask(cvt_color_space(frame, from_space, colorspace), colorspace)
    mask[:int(y), :] = 0
    mask[int(y+h):, :] = 0
    mask[:, :int(x)] = 0
    mask[:, int(x+w):] = 0
    thresholded[mask] += 255
    return thresholded

# SLIDERS/TRACKBARS UTILITIES

def components(colorspace):
    if colorspace == 'ycr_cb':
        return ['y', 'cr', 'cb']
    else: return list(colorspace)


def slider_names(colorspace):
    return [a + ' ' + b for a,b in itertools.product(components(colorspace), ['min', 'max'])]


def init_sliders(colorspace, winname):
    def nothing(val):
        pass

    names = slider_names(colorspace)

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 600,600)
    fn = nothing

    for name in names:
        comps = components(colorspace)
        sw = {
            comps[0]: 0,
            comps[1]: 1,
            comps[2]: 2
        }
        tmp = _hand_thresh_[colorspace][sw[name.split()[0]]][0]  
        cv2.createTrackbar(name, winname, 
        _hand_thresh_[colorspace][sw[name.split()[0]]][0] if name.endswith('min') else _hand_thresh_[colorspace][sw[name.split()[0]]][1], 
        256, fn)


def read_sliders(colorspace, winname):
    res = {name: [] for name in components(colorspace)}
    for name in slider_names(colorspace):
        res[name.split()[0]].append(cv2.getTrackbarPos(name, winname))
    return res

def main(colorspace=None):
    # Here comes the code copy-paste from "detect_multithreaded.py"
    
    # ARGS (standard taken from the original file)
    video_source = 0
    width=300
    height = 200
    num_workers = 4
    score_thresh = 0.2

    ####### Implementing locally worker function ######### 
    def worker(input_q, output_q, output_boxes, cap_params, frame_processed):
        print(">> loading frozen model for worker")
        detection_graph, sess = load_inference_graph()
        sess = tf.Session(graph=detection_graph)
        while True:
            #print("> ===== in worker loop, frame ", frame_processed)
            frame = input_q.get()
            if (frame is not None):
                # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                # while scores contains the confidence for each of these boxes.
                # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

                boxes, scores = detect_objects(
                    frame, detection_graph, sess)
                # draw bounding boxes
                draw_box_on_image(
                    1, cap_params["score_thresh"],
                    scores, boxes, cap_params['im_width'], cap_params['im_height'],
                    frame)
                # add frame annotated with bounding box to queue
                output_q.put(frame)
                output_boxes.put(boxes)
                frame_processed += 1
            else:
                output_q.put(frame)
                output_boxes.put(None)
        sess.close()
    
    input_q = Queue(maxsize=5)
    output_q = Queue(maxsize=5)
    output_boxes = Queue(maxsize=5)
    kernel = np.ones((5,5), np.uint8)

    video_capture = WebcamVideoStream(
        src=video_source, width=width, height=height).start()

    
    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh
    cap_params['num_hands_detect'] = 2

    # spin up workers to paralleize detection
    # FP: note that I added the 'output_boxes' argument to worker's parameters
    pool = Pool(
        num_workers, 
        worker,
        (
            input_q, 
            output_q, 
            output_boxes, 
            cap_params, 
            frame_processed
            )
        )
    
    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Multi-Threaded Detection', 600,600)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()
        out_boxes = output_boxes.get()
        im_width, im_height, ch = output_frame.shape 
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        if (output_frame is not None):
            cv2.imshow('Multi-Threaded Detection', output_frame)

            segm_frame = np.zeros_like(output_frame)
            if out_boxes is not None:

                x,y,w,h = get_rect_points(out_boxes, im_width, im_height)
                segm_frame = threshold_rect(output_frame,colorspace,x,y,w,h)
                segm_frame = cv2.morphologyEx(segm_frame, cv2.MORPH_OPEN, kernel)
            cv2.imshow('Hand-thresholded Image', segm_frame)
        
        if (cv2.waitKey(1) % 256) == ord('q'):
            break
    
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Label map utility functions."""

import logging

import tensorflow as tf
from google.protobuf import text_format
from protos import string_int_label_map_pb2


def _validate_label_map(label_map):
    """Checks if a label map is valid.

    Args:
      label_map: StringIntLabelMap to validate.

    Raises:
      ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.

    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.

    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    """Loads label map proto and returns categories list compatible with eval.

    This function loads a label map and returns a list of dicts, each of which
    has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    We only allow class into the list if its id-label_id_offset is
    between 0 (inclusive) and max_num_classes (exclusive).
    If there are several items mapping to the same id in the label map,
    we will only keep the first one in the categories list.

    Args:
      label_map: a StringIntLabelMapProto or None.  If None, a default categories
        list is created with max_num_classes categories.
      max_num_classes: maximum number of (consecutive) label indices to include.
      use_display_name: (boolean) choose whether to load 'display_name' field
        as category name.  If False or if the display_name field does not exist,
        uses 'name' field as category names instead.
    Returns:
      categories: a list of dictionaries representing all possible categories.
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories


def load_labelmap(path):
    """Loads label map proto.

    Args:
      path: path to StringIntLabelMap proto text file.
    Returns:
      a StringIntLabelMapProto
    """
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map


def get_label_map_dict(label_map_path):
    """Reads a label map and returns a dictionary of label names to id.

    Args:
      label_map_path: path to label_map.

    Returns:
      A dictionary mapping label names to id.
    """
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.name] = item.id
    return label_map_dict


if __name__ == '__main__':

    detection_graph = tf.Graph()

    categories = convert_label_map_to_categories(
        label_map, 
        max_num_classes=NUM_CLASSES, 
        use_display_name=True)
    category_index = create_category_index(categories)

    label_map = load_labelmap(PATH_TO_LABELS)
    _hand_thresh_ = load_thresh()
    colorspace = 'ycr_cb'

    main(colorspace)
    
    

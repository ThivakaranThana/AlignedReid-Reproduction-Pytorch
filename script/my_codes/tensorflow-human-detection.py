# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image


#from script.my_codes.nonmax_suppression import nms


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()
        
def nms(boxes, overlap_threshold=0.2, mode='union'):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick

if __name__ == "__main__":
    model_path = '../../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('Athar.avi')

    img = cv2.imread('my_images/two_persons.jpg')
    #r, img = cap.read()
    #img = cv2.resize(img, (1280, 720))


    boxes, scores, classes, num = odapi.processFrame(img)

    nms_input = np.empty((len(boxes),5))
    nms_input[:, 0] = [i[1] for i in boxes]
    nms_input[:, 1] = [i[0] for i in boxes]
    nms_input[:, 2] = [i[3] for i in boxes]
    nms_input[:, 3] = [i[2] for i in boxes]
    nms_input[:, 4] = scores

    picks_from_nms = nms(nms_input)

    # Visualization of the results of a detection.

    #for i in range(len(boxes)):
        # Class 1 represents human
     #   if classes[i] == 1 and scores[i] > threshold:
      #      box = boxes[i]
            #cv2.circle(img, (box[1], box[0]), 5, (0, 255, 0), -1)
            #cv2.circle(img, (box[3], box[2]), 5, (255, 0, 0), -1)
       #    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

    for i in picks_from_nms:
        if classes[i]==1 and scores[i] > threshold:
            box=boxes[i]
            person_bounding_box =img[box[0]:box[2], box[1]:box[3]]
            cv2.imwrite("person"+str(i)+".jpg",person_bounding_box)
            #cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            
    #cv2.imshow("preview", img)
    cv2.imwrite("boundingbox.jpg", img)
    key = cv2.waitKey(0)

    # while True:
    #     r, img = cap.read()
    #     img = cv2.resize(img, (1280, 720))
    #
    #     boxes, scores, classes, num = odapi.processFrame(img)
    #
    #     # Visualization of the results of a detection.
    #
    #     for i in range(len(boxes)):
    #         # Class 1 represents human
    #         if classes[i] == 1 and scores[i] > threshold:
    #             box = boxes[i]
    #             cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
    #
    #     cv2.imshow("preview", img)
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == ord('q'):
    #         break


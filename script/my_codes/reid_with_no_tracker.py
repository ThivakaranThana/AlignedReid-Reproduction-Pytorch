import dlib
import threading
import numpy as np
import tensorflow as tf
import time
import paramiko
import os,sys,cv2,random,datetime
from reid_per_query import find_matching_id
from scp import SCPClient, SCPException, put, get,asbytes
#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


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

        print("Elapsed Time:", end_time - start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1] * im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3] * im_width))

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

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
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
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


if __name__ == '__main__':

    model_path = '../../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    capture = cv2.VideoCapture('custom_video/test2.mp4')
    frameCounter = 0
    f = open("result", "w")

    try:
        while True:
            # Retrieve the latest image from the webcam
            rc, fullSizeBaseImage = capture.read()
            # height, width, channels = fullSizeBaseImage.shape
            # print height, width
             # Resize the image to 320x240
            # (h,w) = fullSizeBaseImage.shape[:2]
            # M=cv2.getRotationMatrix2D((w/2,h/2),-90,1)
            # baseImage=cv2.warpAffine(fullSizeBaseImage,M,(h,w))
            #baseImage = cv2.resize(fullSizeBaseImage, (1280, 720))
            baseImage=fullSizeBaseImage

            # Check if a key was pressed and if it was Q, then break
            # from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break
            frameCounter += 1
            # Every 10 frames, we will have to determine which faces
            # are present in the frame
            if (frameCounter % 50) == 0:
                boxes, scores, classes, num = odapi.processFrame(baseImage)

                nms_input = np.empty((len(boxes), 5))

                # print boxes[:, 1]

                # nms_input[:, :-1] = boxes
                nms_input[:, 0] = [row[1] for row in boxes]
                nms_input[:, 1] = [row[0] for row in boxes]
                nms_input[:, 2] = [row[3] for row in boxes]
                nms_input[:, 3] = [row[2] for row in boxes]
                nms_input[:, 4] = scores

                picks_from_nms = nms(nms_input)

                for i in picks_from_nms:
                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person_bounding_box = baseImage[box[0]:box[2], box[1]:box[3]]
                        image = "person" + str(i)+'_frameNo_'+str(frameCounter) + '.jpg'
                        cv2.imwrite("query/person" + str(i)+'_frameNo_'+str(frameCounter) +'.jpg', person_bounding_box)
                        Id = find_matching_id(str(image))
                        f.write(Id+"\n")



    except KeyboardInterrupt as e:
        pass

    # Destroy any OpenCV windows and exit the application

exit(0)




#Import the OpenCV and dlib libraries
import dlib
import threading
import numpy as np
import tensorflow as tf
import cv2
import time
import socket
import struct
import pickle
from PIL import Image
from scipy.spatial import distance as dist
from collections import OrderedDict


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


#We are not doing really face recognition
def doRecognizePerson(faceNames, fid):
    time.sleep(2)
    faceNames[ fid ] = "Person " + str(fid)


if __name__ == '__main__':

    model_path = '../../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    # Open the first webcame device
    capture = cv2.VideoCapture('Athar.avi')
    # Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    # Start the window thread for the two windows we are using
    cv2.startWindowThread()

    # The color of the rectangle we draw around the face
    rectangleColor = (0, 165, 255)

    # variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    # Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}

    try:
        n=1
        while True:
            # Retrieve the latest image from the webcam
            rc, fullSizeBaseImage = capture.read()

             # Resize the image to 320x240
            baseImage = cv2.resize(fullSizeBaseImage, (1280, 720))

            # Check if a key was pressed and if it was Q, then break
            # from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break

            # Result image is the image we will show the user, which is a
            # combination of the original image from the webcam and the
            # overlayed rectangle for the largest face
            #resultImage = baseImage.copy()

                # STEPS:
                # * Update all trackers and remove the ones that are not
                #   relevant anymore
                # * Every 10 frames:
                #       + Use face detection on the current frame and look
                #         for faces.
                #       + For each found face, check if centerpoint is within
                #         existing tracked box. If so, nothing to do
                #       + If centerpoint is NOT in existing tracked box, then
                #         we add a new tracker with a new face-id

                # Increase the framecounter
            frameCounter += 1

                # Update all the trackers and remove the ones for which the update
                # indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[fid].update(baseImage)

                # If the tracking quality is good enough, we must delete
                # this tracker
                if trackingQuality < 7:
                    fidsToDelete.append(fid)

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop(fid, None)

            # Every 10 frames, we will have to determine which faces
            # are present in the frame
            if (frameCounter % 10) == 0:
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

                persons = []
                for i in picks_from_nms:
                    if classes[i] == 1 and scores[i] > threshold:
                        persons.append(boxes[i])



                # For the face detection, we need to make use of a gray
                    # colored image so we will convert the baseImage to a
                    # gray-based image
    #                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                    # Now use the haar cascade detector to find all faces
                    # in the image
     #               faces = faceCascade.detectMultiScale(gray, 1.3, 5)

                    # Loop over all faces and check if the area for this
                    # face is the largest so far
                    # We need to convert it to int here because of the
                    # requirement of the dlib tracker. If we omit the cast to
                    # int here, you will get cast errors since the detector
                    # returns numpy.int32 and the tracker requires an int

                for (y1, x1, y2, x2) in persons:
                    # calculate the centerpoint
                    x_bar = (x1+ x2)/2.0
                    y_bar = (y1 +y2)/2.0

                    # Variable holding information which faceid we
                    # matched with
                    matchedFid = None

                    # Now loop over all the trackers and check if the
                    # centerpoint of the face is within the box of a
                    # tracker
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        # check if the centerpoint of the face is within the
                        # rectangleof a tracker region. Also, the centerpoint
                        # of the tracker region must be within the region
                        # detected as a face. If both of these conditions hold
                        # we have a match
                        if ((t_x <= x_bar <= (t_x + t_w)) and
                                (t_y <= y_bar <= (t_y + t_h)) and
                                (x1<= t_x_bar <= x2) and
                                (y1 <= t_y_bar <= y2)):
                            matchedFid = fid

                    # If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        print("Creating new tracker " + str(currentFaceID))

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle(x1- 10,
                                                           y1 - 20,
                                                           x2 + 10,
                                                           y2 + 20))

                        faceTrackers[currentFaceID] = tracker

                        # Start a new thread that is used to simulate
                        # face recognition. This is not yet implemented in this
                        # version :)
                        t = threading.Thread(target=doRecognizePerson,
                                             args=(faceNames, currentFaceID))
                        t.start()

                        # Increase the currentFaceID counter
                        currentFaceID += 1

                # Now loop over all the trackers we have and draw the rectangle
                # around the detected faces. If we 'know' the name for this person
                # (i.e. the recognition thread is finished), we print the name
                # of the person, otherwise the message indicating we are detecting
                # the name of the person
            for fid in faceTrackers.keys():
                tracked_position = faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(baseImage, (t_x, t_y),
                              (t_x + t_w, t_y + t_h),
                              rectangleColor, 2)

                if fid in faceNames.keys():
                    cv2.putText(baseImage, faceNames[fid],
                                (int(t_x + t_w / 2), int(t_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(baseImage, "Detecting...",
                                (int(t_x + t_w / 2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

            # Since we want to show something larger on the screen than the
                # original 320x240, we resize the image again
                #
                # Note that it would also be possible to keep the large version
                # of the baseimage and make the result image a copy of this large
                # base image and use the scaling factor to draw the rectangle
                # at the right coordinates.
                largeResult = cv2.resize(baseImage,
                                         (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

                # Finally, we want to show the images on the screen
            #cv2.imshow("base-image", baseImage)
            #cv2.imshow("result-image", largeResult)
            cv2.imwrite("result"+str(n)+".jpg",largeResult)
            n=n+1




        # To ensure we can also deal with the user pressing Ctrl-C in the console
        # we have to check for the KeyboardInterrupt exception and break out of
        # the main loop
    except KeyboardInterrupt as e:
        pass

    # Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)




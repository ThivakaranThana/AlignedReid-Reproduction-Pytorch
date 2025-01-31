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
import paramiko
import os,sys,cv2,random,datetime
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


#We are not doing really face recognition
def doRecognizePerson(faceNames, fid):
    time.sleep(2)
    faceNames[ fid ] = "Person " + str(fid)


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


if __name__ == '__main__':

    model_path = '../../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    # connect to socket
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.connect(('192.168.8.100', 8485))
    # connection = client_socket.makefile('wb')
    # Open the first webcame device
    capture = cv2.VideoCapture('custom_video/vedha.mov')

    # Create two opencv named windows
    #cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    #cv2.moveWindow("base-image", 0, 100)
    #cv2.moveWindow("result-image", 400, 100)

    # Start the window thread for the two windows we are using
    #cv2.startWindowThread()

    # The color of the rectangle we draw around the face
    rectangleColor = (0, 165, 255)

    # variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    # Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}

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
            # pressedKey = cv2.waitKey(2)
            # if pressedKey == ord('Q'):
            #     break

            # Result image is the image we will show the user, which is a
            # combination of the original image from the webcam and the
            # overlayed rectangle for the largest face
            resultImage = baseImage.copy()

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
                # for i in range(len(boxes)):
                #     box = boxes[i]
                #     person_bounding_box2 = baseImage[box[0]:box[2], box[1]:box[3]]
                #     cv2.imwrite("query2/person_" + str(frameCounter) + "_" + str(i) + ".jpg", person_bounding_box2)

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
                count = 0
                for (y1, x1, y2, x2) in persons:
                    # person_bounding_box1 = resultImage[y1:y2, x1:x2]
                    # cv2.imwrite("query1/person_" + str(frameCounter) + str(count)+".jpg", person_bounding_box1)
                    # count = count +1
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
                            person_bounding_box = resultImage[y1:y2, x1:x2]
                            image = faceNames[fid] + "_frame_no" + str(frameCounter) + ".jpg";
                            cv2.imwrite("query/" + str(image), person_bounding_box)


                    # If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        print("Creating new tracker " + str(currentFaceID))

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        # tracker.start_track(baseImage,
                        #                       dlib.rectangle(x1-10,
                        #                                     y1 - 30,
                        #                                     x2 + 10,
                        #                                     y2 + 30))
                        tracker.start_track(baseImage,
                                            dlib.rectangle(x1,
                                                           y1 ,
                                                           x2 ,
                                                           y2))

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

            # for fid in faceTrackers.keys():
            #     tracked_position = faceTrackers[fid].get_position()
            #
            #     t_x = int(tracked_position.left())
            #     t_y = int(tracked_position.top())
            #     t_w = int(tracked_position.width())
            #     t_h = int(tracked_position.height())
            #
            #     # cv2.rectangle(resultImage, (t_x, t_y),
            #     #               (t_x + t_w, t_y + t_h),
            #     #               rectangleColor, 2)
            #     # person_bounding_box = resultImage[t_y:(t_y+t_h), t_x:(t_x+t_w)]
            #     # cv2.imshow("person", person_bounding_box)
            #     if fid in faceNames.keys():
            #         # cv2.putText(resultImage, faceNames[fid],
            #         #              (int(t_x + t_w / 2), int(t_y)),
            #         #          cv2.FONT_HERSHEY_SIMPLEX,
            #         #              0.5, (255, 255, 255), 2)
            #         person_bounding_box = resultImage[t_y:(t_y + t_h+50), t_x:(t_x + t_w+20)]
            #         if (frameCounter % 10) == 0:
            #             if (t_x > 0) & (t_y > 0):
            #                 image = faceNames[fid]+"_frame_no"+ str(frameCounter)+".jpg";
            #                 cv2.imwrite("query/"+str(image), person_bounding_box)
            #                 # ssh = createSSHClient("10.12.67.36", 22, "madhushanb", "group10@fyp")
            #                 # scp = SCPClient(ssh.get_transport())
            #                 # scp.put("query/"+str(image), '/home/madhushanb/sphereface/bounding_Box_ID', True)
            #         # result, frame = cv2.imencode('.jpg', person_bounding_box, encode_param)
            #         # data = pickle.dumps(frame, 0)
            #         # size = len(data)
            #         #
            #         # print("{}".format(size))
            #         # client_socket.sendall(struct.pack(">L", size) + data)
            #         # print "finished sending bounding box"
            #         # data1 = pickle.dumps(faceNames[fid], 0)
            #         # size1 = len(data1)
            #         # print("{}".format(size1))
            #         # client_socket.sendall(struct.pack(">L", size1) + data1)
            #         # print "finshed sending name of the bounding box"
            #         # data2 = pickle.dumps(frameCounter, 0)
            #         # size2 = len(data2)
            #         # print("{}".format(size2))
            #         # client_socket.sendall(struct.pack(">L", size2) + data2)
            #         # print "finshed sending frame no of the bounding box"
            #
            #
            #     else:
            #
            #          cv2.putText(resultImage, "Detecting...",
            #                      (int(t_x + t_w / 2), int(t_y)),
            #                      cv2.FONT_HERSHEY_SIMPLEX,
            #                      0.5, (255, 255, 255), 2)

            # Since we want to show something larger on the screen than the
                # original 320x240, we resize the image again
                #
                # Note that it would also be possible to keep the large version
                # of the baseimage and make the result image a copy of this large
                # base image and use the scaling factor to draw the rectangle
                # at the right coordinates.
                #largeResult = cv2.resize(resultImage,
#                                         (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

                # Finally, we want to show the images on the screen
            #cv2.imshow("base-image", baseImage)
            #cv2.imshow("result-image", resultImage)




        # To ensure we can also deal with the user pressing Ctrl-C in the console
        # we have to check for the KeyboardInterrupt exception and break out of
        # the main loop
    except AttributeError:
        pass


    # Destroy any OpenCV windows and exit the application
    #cv2.destroyAllWindows()
    exit(0)




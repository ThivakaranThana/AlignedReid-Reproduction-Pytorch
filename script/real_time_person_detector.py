# Import the OpenCV and dlib libraries
import cv2
import dlib
import numpy as np
import threading
import time
from imutils.object_detection import non_max_suppression
import imutils

# The desired output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600


# We are not doing really face recognition
def doRecognizePerson(personNames, personID):
    time.sleep(2)
    personNames[personID] = "Person " + str(personID)


def detectAndTrackMultiplePersons():
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Open the first webcame device
    capture = cv2.VideoCapture(0)

    # Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    # Start the window thread for the two windows we are using
    # cv2.startWindowThread()

    # The color of the rectangle we draw around the face
    rectangleColor = (0, 165, 255)

    # variables holding the current frame number and the current faceid
    frameCounter = 0
    currentPersonID = 0

    # Variables holding the correlation trackers and the name per faceid
    personTrackers = {}
    faceNames = {}

    try:
        while True:
            # Retrieve the latest image from the webcam
            rc, baseImage = capture.read()
            baseImage = imutils.resize(baseImage, width=min(400, baseImage.shape[1]))

            # Resize the image to 320x240
            # baseImage = cv2.resize(fullSizeBaseImage, (320, 240))

            # Check if a key was pressed and if it was Q, then break
            # from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break

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
            personIDsToDelete = []
            for ID in personTrackers.keys():
                trackingQuality = personTrackers[ID].update(baseImage)

                # If the tracking quality is good enough, we must delete
                # this tracker
                if trackingQuality < 7:
                    personIDsToDelete.append(ID)

            for ID in personIDsToDelete:
                print("Removing ID " + str(ID) + " from list of trackers")
                personTrackers.pop(ID, None)

            # Every 10 frames, we will have to determine which faces
            # are present in the frame
            if (frameCounter % 10) == 0:

                # For the face detection, we need to make use of a gray
                # colored image so we will convert the baseImage to a
                # gray-based image
                # gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                # Now use the haar cascade detector to find all faces
                # in the image
                # faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                # detect people in the image
                (rects, weights) = hog.detectMultiScale(baseImage, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)
                # apply non-maxima suppression to the bounding boxes using a
                # fairly large overlap threshold to try to maintain overlapping
                # boxes that are still people
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

                pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

                # Loop over all faces and check if the area for this
                # face is the largest so far
                # We need to convert it to int here because of the
                # requirement of the dlib tracker. If we omit the cast to
                # int here, you will get cast errors since the detector
                # returns numpy.int32 and the tracker requires an int
                for (_x, _y, _x2, _y2) in pick:
                    x = int(_x)
                    y = int(_y)
                    w = int((_x+_x2)/2)
                    h = int((_y+_y2)/2)

                    # calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    # Variable holding information which faceid we
                    # matched with
                    matchedPersonID = None

                    # Now loop over all the trackers and check if the
                    # centerpoint of the face is within the box of a
                    # tracker
                    for ID in personTrackers.keys():
                        tracked_position = personTrackers[ID].get_position()

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
                                (x <= t_x_bar <= (x + w)) and
                                (y <= t_y_bar <= (y + h))):
                            matchedPersonID = ID

                    # If no matched ID, then we have to create a new tracker
                    if matchedPersonID is None:
                        print("Creating new tracker " + str(currentPersonID))

                        # Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle(x - 10,
                                                           y - 20,
                                                           x + w + 10,
                                                           y + h + 20))

                        personTrackers[currentPersonID] = tracker

                        # Start a new thread that is used to simulate
                        # face recognition. This is not yet implemented in this
                        # version :)
                        t = threading.Thread(target=doRecognizePerson,
                                             args=(faceNames, currentPersonID))
                        t.start()

                        # Increase the currentPersonID counter
                        currentPersonID += 1

            # Now loop over all the trackers we have and draw the rectangle
            # around the detected faces. If we 'know' the name for this person
            # (i.e. the recognition thread is finished), we print the name
            # of the person, otherwise the message indicating we are detecting
            # the name of the person
            for ID in personTrackers.keys():
                tracked_position = personTrackers[ID].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(resultImage, (t_x, t_y),
                              (t_x + t_w, t_y + t_h),
                              rectangleColor, 2)

                if ID in faceNames.keys():
                    cv2.putText(resultImage, faceNames[ID],
                                (int(t_x + t_w / 2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting...",
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
            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

            # Finally, we want to show the images on the screen
            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)

    # To ensure we can also deal with the user pressing Ctrl-C in the console
    # we have to check for the KeyboardInterrupt exception and break out of
    # the main loop
    except KeyboardInterrupt as e:
        pass

    # Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    detectAndTrackMultiplePersons()
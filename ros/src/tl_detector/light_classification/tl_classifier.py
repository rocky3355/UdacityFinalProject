import cv2
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.create_blob_detector()
        self.kernel = np.ones((5, 5), np.float32) / 25


    def create_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = False
        params.filterByColor = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByCircularity = True
        params.minCircularity = 0.8

        version = cv2.__version__.split('.')
        is_old_cv_version = int(version[0]) < 3

        if is_old_cv_version:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)


    # TODO: Add some kind of mean filter?
    def get_classification(self, image):
        mask_red = cv2.inRange(image, (0, 0, 100), (50, 50, 255))
        mask_yellow = cv2.inRange(image, (20, 200, 200), (60, 255, 255))
        mask = cv2.bitwise_or(mask_red, mask_yellow)

        masked = cv2.bitwise_and(image, image, mask=mask)
        blurred = cv2.filter2D(masked, -1, self.kernel)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        keypoints = self.detector.detect(gray)
        if len(keypoints) > 0:
            #print('RED')
            return TrafficLight.RED

        #print('GREEN or UNKNOWN')
        return TrafficLight.UNKNOWN



#mask_red = cv2.inRange(image, (0, 0, 100), (50, 50, 255))
#mask_yellow = cv2.inRange(image, (20, 200, 200), (50, 255, 255))

#masked_red = cv2.bitwise_and(image, image, mask=mask_red)
#masked_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)

#blurred_red = cv2.filter2D(masked_red, -1, self.kernel)
#blurred_yellow = cv2.filter2D(masked_yellow, -1, self.kernel)

#gray_red = cv2.cvtColor(blurred_red, cv2.COLOR_BGR2GRAY)
#gray_yellow = cv2.cvtColor(blurred_yellow, cv2.COLOR_BGR2GRAY)

#keypoints_red = self.detector.detect(gray_red)
#if len(keypoints_red) > 0:
#    #print('RED')
#    return TrafficLight.RED

#keypoints_yellow = self.detector.detect(gray_yellow)
#if len(keypoints_yellow) > 0:
#    # print('YELLOW')
#    return TrafficLight.YELLOW

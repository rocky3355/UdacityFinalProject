import cv2
import numpy as np
from styx_msgs.msg import TrafficLight

DETECTION_FILTER_SIZE = 3

class TLClassifier(object):
    def __init__(self):
        self.create_blob_detector()
        self.kernel = np.ones((5, 5), np.float32) / 25#

        self.detection_filter = []
        for i in range(DETECTION_FILTER_SIZE):
            self.detection_filter.append(TrafficLight.UNKNOWN)

        self.detection_filter_idx = 0
        self.detection_filter_current_result = TrafficLight.UNKNOWN


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


    def get_classification(self, image):
        mask_red = cv2.inRange(image, (0, 0, 100), (50, 50, 255))
        mask_yellow = cv2.inRange(image, (20, 200, 200), (60, 255, 255))
        mask = cv2.bitwise_or(mask_red, mask_yellow)

        masked = cv2.bitwise_and(image, image, mask=mask)
        blurred = cv2.filter2D(masked, -1, self.kernel)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        keypoints = self.detector.detect(gray)
        if len(keypoints) > 0:
            self.detection_filter_add(TrafficLight.RED)
        else:
            self.detection_filter_add(TrafficLight.UNKNOWN)

        self.detection_filter_evaluate()
        return self.detection_filter_current_result


    def detection_filter_add(self, value):
        self.detection_filter[self.detection_filter_idx] = value
        self.detection_filter_idx += 1
        if self.detection_filter_idx == DETECTION_FILTER_SIZE:
            self.detection_filter_idx = 0


    def detection_filter_evaluate(self):
        counter = 0
        for detection in self.detection_filter:
            if detection is not self.detection_filter_current_result:
                counter += 1

        if counter == DETECTION_FILTER_SIZE:
            self.detection_filter_current_result = self.detection_filter[0]

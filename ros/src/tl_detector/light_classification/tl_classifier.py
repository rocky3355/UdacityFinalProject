import cv2
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        self.create_blob_detector()


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

        version = (cv2.__version__).split('.')
        is_old_cv_version = int(version[0]) < 3

        if is_old_cv_version:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)


    def get_classification(self, image):
        mask = cv2.inRange(image, (0, 0, 100), (50, 50, 255))
        res = cv2.bitwise_and(image, image, mask=mask)

        kernel = np.ones((5, 5), np.float32) / 25
        res = cv2.filter2D(res, -1, kernel)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        keypoints = self.detector.detect(res)
        if len(keypoints) > 0:
            return TrafficLight.RED

        return TrafficLight.GREEN

import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from scipy import misc
from styx_msgs.msg import TrafficLight


SAVE_IMAGES = True
MODEL_IMG_SIZE = (64, 64)
TRAFFIC_LIGHT_MIN_PROB = 0.6
MODEL_FILE_NAME = 'training/real/model_real.h5'


class TLClassifier(object):
    def __init__(self):
        self.create_blob_detector()
        self.kernel = np.ones((5, 5), np.float32) / 13.0

        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_path, MODEL_FILE_NAME)
        self.model = keras.models.load_model(model_path)
        self.graph = tf.get_default_graph()


    def create_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = False
        params.filterByColor = False
        params.filterByInertia = True
        params.filterByConvexity = False
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.minInertiaRatio = 0.7
        params.maxInertiaRatio = 1.0
        version = cv2.__version__.split('.')
        is_old_cv_version = int(version[0]) < 3

        if is_old_cv_version:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)


    def increase_brightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


    def perform_blob_detection(self, image):
        image = self.increase_brightness(image, 40)
        blurred = cv2.filter2D(image, -1, self.kernel)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        if SAVE_IMAGES:
            misc.imsave('gray.jpg', gray)
        key_points = self.detector.detect(gray)

        if SAVE_IMAGES:
            for marker in key_points:
                image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 0, 255))
            misc.imsave('key_points.jpg', image)

        return key_points


    def get_traffic_light_height(self, key_points):
        min_y = 1000
        max_y = -1
        for kp in key_points:
            if kp.pt[1] < min_y:
                min_y = kp.pt[1]
            if kp.pt[1] > max_y:
                max_y = kp.pt[1]

        height = max_y - min_y
        return height


    #TODO: Rename, names too similar
    def calc_traffic_light_center_and_height(self, key_points):
        center_x = 0
        center_y = 0
        for kp in key_points:
            center_x += kp.pt[0]
            center_y += kp.pt[1]

        center_x /= len(key_points)
        center_y /= len(key_points)
        height = self.get_traffic_light_height(key_points)

        return [center_x, center_y, height]


    def get_traffic_light_center_and_height(self, key_points):
        traffic_light_center_height = [-1, -1, -1]

        for idx, kp in enumerate(key_points):
            traffic_light_kp = [kp]
            for idx2, kp2 in enumerate(key_points):
                if idx == idx2:
                    continue
                # TODO: Also check for max y distance? Or take all points with same x and filter 3 closest ones with
                #       respect to y
                if abs(kp.pt[0] - kp2.pt[0]) < 5:
                    traffic_light_kp.append(kp2)
                if len(traffic_light_kp) == 3:
                    break

            if len(traffic_light_kp) == 3:
                traffic_light_center_height = self.calc_traffic_light_center_and_height(traffic_light_kp)
                break

        return traffic_light_center_height


    def create_window_from_traffic_light_center(self, traffic_light_center_height):
        x = traffic_light_center_height[0]
        y = traffic_light_center_height[1]
        height = traffic_light_center_height[2]
        width = height * 0.7
        # "height" only represents the distance between
        # first and last lamp inside the traffic light
        real_height = height * 1.6

        top_left = [int(x - width / 2), int(y - real_height / 2)]
        bottom_right = [int(top_left[0] + width), int(top_left[1] + real_height)]

        if top_left[0] < 0 or top_left[1] < 0 or bottom_right[0] < 0 or bottom_right[1] < 0:
            return None

        return [top_left, bottom_right]


    def perform_object_detection(self, image, window):
        if window is None:
            return TrafficLight.UNKNOWN

        window_img = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        window_img = cv2.resize(window_img, MODEL_IMG_SIZE)
        if SAVE_IMAGES:
            misc.imsave('model_input.jpg', window_img)

        with self.graph.as_default():
            result = self.model.predict(np.array([window_img]))

        detected_idx = -1
        max_prob = result[0][0]

        for idx in range(1, len(result[0])):
            prob = result[0][idx]
            if prob > TRAFFIC_LIGHT_MIN_PROB and prob > max_prob:
                max_prob = prob
                detected_idx = idx

        if detected_idx == 1:
            return TrafficLight.GREEN
        elif detected_idx == 2:
            # Yellow will also be returned as red
            return TrafficLight.RED
        elif detected_idx == 3:
            return TrafficLight.RED
        return TrafficLight.UNKNOWN


    def get_classification(self, image):
        traffic_light_detection = TrafficLight.UNKNOWN
        key_points = self.perform_blob_detection(image)
        print('# Key points: ' + str(len(key_points)))

        if len(key_points) >= 3:
            traffic_light_center_height = self.get_traffic_light_center_and_height(key_points)
            if traffic_light_center_height[0] > -1:
                window = self.create_window_from_traffic_light_center(traffic_light_center_height)
                traffic_light_detection = self.perform_object_detection(image, window)

        print('TL result: ' + str(traffic_light_detection))
        return traffic_light_detection



classifier = TLClassifier()
img = misc.imread('training/real/source/images/image_0.jpg')
classification = classifier.get_classification(img)

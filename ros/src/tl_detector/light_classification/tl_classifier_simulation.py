import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from scipy import misc
from styx_msgs.msg import TrafficLight

IS_TEST = True
MODEL_IMG_SIZE = (64, 64)
TRAFFIC_LIGHT_MIN_PROB = 0.5
MODEL_FILE_NAME = 'training/simulation/model_simulation.h5'


class TLClassifierSimulation(object):
    def __init__(self):
        self.detector = self.create_blob_detector()
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
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        return detector


    def adjust_brightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            value = -value
            lim = value
            v[v <= lim] = 0
            v[v > lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


    def perform_blob_detection(self, image):
        image = self.adjust_brightness(image, 40)
        if IS_TEST:
            misc.imsave('sim_brightness.jpg', image)

        image = cv2.filter2D(image, -1, self.kernel)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if IS_TEST:
            misc.imsave('sim_gray.jpg', image)

        key_points = self.detector.detect(image)

        if IS_TEST:
            for marker in key_points:
                image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 0, 255))
            misc.imsave('sim_key_points.jpg', image)

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


    def map_detected_index_to_traffic_light(self, detected_idx):
        if detected_idx == 1:
            return TrafficLight.GREEN
        elif detected_idx == 2:
            # Yellow will also be returned as red
            return TrafficLight.RED
        elif detected_idx == 3:
            return TrafficLight.RED
        return TrafficLight.UNKNOWN


    def perform_object_detection(self, image, window):
        if window is None:
            return TrafficLight.UNKNOWN

        window_img = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        window_img = cv2.resize(window_img, MODEL_IMG_SIZE)
        if IS_TEST:
            misc.imsave('sim_model_input.jpg', window_img)

        with self.graph.as_default():
            prediction = self.model.predict(np.array([window_img]))

        detected_idx = -1
        max_prob = prediction[0][0]

        for idx in range(1, len(prediction[0])):
            prob = prediction[0][idx]
            if prob > TRAFFIC_LIGHT_MIN_PROB and prob > max_prob:
                max_prob = prob
                detected_idx = idx

        traffic_light_detection = self.map_detected_index_to_traffic_light(detected_idx)
        return traffic_light_detection


    def get_classification(self, image):
        traffic_light_detection = TrafficLight.UNKNOWN

        key_points = self.perform_blob_detection(image)
        if len(key_points) >= 3:
            traffic_light_center_height = self.get_traffic_light_center_and_height(key_points)
            if traffic_light_center_height[0] > -1:
                window = self.create_window_from_traffic_light_center(traffic_light_center_height)
                traffic_light_detection = self.perform_object_detection(image, window)

        return traffic_light_detection


if IS_TEST:
    classifier = TLClassifierSimulation()
    img = misc.imread('training/simulation/source/images/image_0.jpg')
    classifier.get_classification(img)

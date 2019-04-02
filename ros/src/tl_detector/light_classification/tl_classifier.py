import os
import cv2
#import keras
import numpy as np
#import tensorflow as tf
from scipy import misc
#from styx_msgs.msg import TrafficLight


IS_SIMULATION = False
SAVE_IMAGES = True
MODEL_IMG_SIZE = (64, 64)
TRAFFIC_LIGHT_MIN_PROB = 0.6
MODEL_FILE_NAME = 'training/real/model_real.h5'


class TLClassifier(object):
    def __init__(self):
        self.create_blob_detector()
        self.kernel = np.ones((5, 5), np.float32)

        if IS_SIMULATION:
            self.kernel /= 13.0
        else:
            self.kernel /= 35.0

        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_path, MODEL_FILE_NAME)
        #self.model = keras.models.load_model(model_path)
        #self.graph = tf.get_default_graph()

        #self.windows = self.create_search_windows()


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
        if IS_SIMULATION:
            image = self.adjust_brightness(image, 40)
        else:
            image = self.adjust_brightness(image, -30)

        image = cv2.filter2D(image, -1, self.kernel)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if SAVE_IMAGES:
            misc.imsave('gray.jpg', image)

        key_points = self.detector.detect(image)

        if SAVE_IMAGES:
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
        if SAVE_IMAGES:
            misc.imsave('sim_model_input.jpg', window_img)

        with self.graph.as_default():
            result = self.model.predict(np.array([window_img]))

        detected_idx = -1
        max_prob = result[0][0]

        for idx in range(1, len(result[0])):
            prob = result[0][idx]
            if prob > TRAFFIC_LIGHT_MIN_PROB and prob > max_prob:
                max_prob = prob
                detected_idx = idx

        traffic_light_detection = self.map_detected_index_to_traffic_light(detected_idx)
        return traffic_light_detection


    def slide_window(self, img_shape, x_start_stop, y_start_stop, xy_window, xy_overlap):
        img_width = img_shape[0]
        img_height = img_shape[1]

        x_start = x_start_stop[0]
        x_stop = x_start_stop[1]
        y_start = y_start_stop[0]
        y_stop = y_start_stop[1]

        if x_start is None:
            x_start = 0
        if x_stop is None:
            x_stop = img_width
        if y_start is None:
            y_start = 0
        if y_stop is None:
            y_stop = img_height

        x_span = x_stop - x_start
        y_span = y_stop - y_start

        x_step = xy_window[0] * (1 - xy_overlap[0])
        y_step = xy_window[1] * (1 - xy_overlap[1])

        x_windows = np.int(x_span / x_step)
        y_windows = np.int(y_span / y_step)

        window_list = []
        for y in range(y_windows):
            for x in range(x_windows):
                left_top = (np.int(x_start + x * x_step), np.int(y_start + y * y_step))
                right_bottom = (np.int(left_top[0] + xy_window[0]), np.int(left_top[1] + xy_window[1]))
                if right_bottom[0] <= img_width and right_bottom[1] <= img_height:
                    window_list.append((left_top, right_bottom))

        return window_list


    def create_search_windows(self, img_shape):
        overlap = (0.5, 0.5)
        near_windows = self.slide_window(img_shape, (100, 700), (0, 200), (50, 160), overlap)
        mid_windows = self.slide_window(img_shape, (200, 600), (0, 200), (40, 120), overlap)
        far_windows = self.slide_window(img_shape, (300, 500), (0, 200), (20, 60), overlap)
        self.windows = near_windows + mid_windows + far_windows


    def get_window_images(self, image):
        # idx = 0
        window_images = []
        for window in self.windows:
            window_image = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            window_image = cv2.resize(window_image, MODEL_IMG_SIZE)
            # misc.imsave('boxes/' + str(idx) + '.jpg', window_image)
            # idx += 1
            window_images.append(window_image)
            # cv2.rectangle(image, window[0], window[1], (0, 0, 255), 6)

        # misc.imsave('test.jpg', image)
        # exit(0)
        window_images = np.array(window_images)
        return window_images


    def create_blob_detector2(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.filterByColor = False
        #params.filterByInertia = True
        params.filterByConvexity = False
        params.filterByCircularity = False
        params.minArea = 300
        params.maxArea = 2000
        #params.minInertiaRatio = 0
        #params.maxInertiaRatio = 1.0

        version = cv2.__version__.split('.')
        is_old_cv_version = int(version[0]) < 3

        if is_old_cv_version:
            return cv2.SimpleBlobDetector(params)

        return cv2.SimpleBlobDetector_create(params)


    def get_classification(self, image):
        #traffic_light_detection = TrafficLight.UNKNOWN

        if IS_SIMULATION:
            key_points = self.perform_blob_detection(image)
            if len(key_points) >= 3:
                traffic_light_center_height = self.get_traffic_light_center_and_height(key_points)
                if traffic_light_center_height[0] > -1:
                    window = self.create_window_from_traffic_light_center(traffic_light_center_height)
                    traffic_light_detection = self.perform_object_detection(image, window)
        else:
            image = image[:int(image.shape[0] / 2), :]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if SAVE_IMAGES:
                misc.imsave('real_gray.jpg', image)
            mask_bright = cv2.inRange(image, 240, 255)
            image = cv2.bitwise_and(image, mask_bright)
            if SAVE_IMAGES:
                misc.imsave('real_masked.jpg', image)
            image = cv2.filter2D(image, -1, self.kernel)
            if SAVE_IMAGES:
                misc.imsave('real_blurred.jpg', image)
            detector2 = self.create_blob_detector2()
            key_points = detector2.detect(image)
            if SAVE_IMAGES:
                for marker in key_points:
                    image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 0, 255))
                misc.imsave('real_key_points.jpg', image)
            print(len(key_points))

            window_images = self.get_window_images(key_points)
            with self.graph.as_default():
                result = self.model.predict(window_images)



        print('TL result: ' + str(traffic_light_detection))
        return traffic_light_detection



classifier = TLClassifier()
img = misc.imread('training/real/source/images/image_136.jpg')
classification = classifier.get_classification(img)

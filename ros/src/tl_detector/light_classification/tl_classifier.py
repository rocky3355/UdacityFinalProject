import os
import cv2
import keras
import rospy
import numpy as np
import tensorflow as tf
from scipy import misc
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from styx_msgs.msg import TrafficLight


SAVE_IMAGES = False
IS_SIMULATION = False
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
            self.kernel /= 100.0

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


    def slide_window(self, img_width, img_height, x_center, y_start_stop, window_size, xy_overlap):
        y_start = y_start_stop[0]
        y_stop = y_start_stop[1]
        half_window_width = np.int(window_size[0] / 2)
        window_height = window_size[1]

        if y_start is None:
            y_start = 0
        if y_stop is None:
            y_stop = img_height

        y_span = y_stop - y_start
        y_step = window_height * (1 - xy_overlap[1])
        y_windows = np.int(y_span / y_step)

        window_list = []
        for y in range(y_windows):
            left_top = (x_center - half_window_width, np.int(y_start + y * y_step))
            right_bottom = (x_center + half_window_width, np.int(left_top[1] + window_height))
            if left_top[0] >= 0 and left_top[0] >= 0 and right_bottom[0] <= img_width and right_bottom[1] <= img_height:
                window_list.append((left_top, right_bottom))

        return window_list


    def create_search_windows(self, traffic_light_x, img_width, img_height):
        windows = []
        overlap = (0.5, 0.5)

        for tlx in traffic_light_x:
            for shift in range(-5, 5, 5):
                near_windows = self.slide_window(img_width, img_height, tlx + shift, (None, None), (50, 160), overlap)
                mid_windows = self.slide_window(img_width, img_height, tlx + shift, (None, None), (20, 80), overlap)
                far_windows = self.slide_window(img_width, img_height, tlx + shift, (None, None), (20, 60), overlap)
                windows += near_windows + mid_windows + far_windows

        return windows


    def get_window_images(self, image, windows):
        # idx = 0
        window_images = []
        for window in windows:
            window_image = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            window_image = cv2.resize(window_image, MODEL_IMG_SIZE)
            window_images.append(window_image)
            # cv2.rectangle(image, window[0], window[1], (0, 0, 255), 6)

        # misc.imsave('test.jpg', image)
        # exit(0)
        window_images = np.array(window_images)
        return window_images


    def create_blob_detector2(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.filterByArea = True
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.minArea = 30
        params.maxArea = 1500

        version = cv2.__version__.split('.')
        is_old_cv_version = int(version[0]) < 3

        if is_old_cv_version:
            return cv2.SimpleBlobDetector(params)

        return cv2.SimpleBlobDetector_create(params)


    def printtl(self, detection):
        if detection == 1:
            print('GREEN')
        elif detection == 2:
            print('YELLOW')
        elif detection == 3:
            print('RED')
        else:
            print('UNKNOWN')


    def get_classification(self, image):
        traffic_light_detection = TrafficLight.UNKNOWN

        if IS_SIMULATION:
            key_points = self.perform_blob_detection(image)
            if len(key_points) >= 3:
                traffic_light_center_height = self.get_traffic_light_center_and_height(key_points)
                if traffic_light_center_height[0] > -1:
                    window = self.create_window_from_traffic_light_center(traffic_light_center_height)
                    traffic_light_detection = self.perform_object_detection(image, window)
        else:
            img_width = image.shape[1]
            img_height = int(image.shape[0] / 2)

            processed_img = image[:img_height, :]
            # TODO: Evaluate image darkness and adjust brightness accordingly
            processed_img = self.adjust_brightness(processed_img, -200)
            if SAVE_IMAGES:
                misc.imsave('real_brightness.jpg', processed_img)
            #processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            #if SAVE_IMAGES:
            #    misc.imsave('real_gray.jpg', processed_img)
            #processed_img = 255 - processed_img
            #mask_bright = cv2.inRange(processed_img, 220, 255)
            #processed_img = cv2.bitwise_and(processed_img, mask_bright)
            #if SAVE_IMAGES:
            #    misc.imsave('real_filtered.jpg', processed_img)
            #processed_img = cv2.filter2D(processed_img, -1, self.kernel)
            #if SAVE_IMAGES:
            #    misc.imsave('real_blurred.jpg', processed_img)
            detector2 = self.create_blob_detector2()
            key_points = detector2.detect(processed_img)

            traffic_light_x = []

            if len(key_points) >= 2:
                for idx, kp in enumerate(key_points):
                    traffic_light_kp = [kp]
                    for idx2, kp2 in enumerate(key_points):
                        if idx == idx2:
                            continue
                        if abs(kp.pt[0] - kp2.pt[0]) < 2:
                            traffic_light_kp.append(kp2)
                        if len(traffic_light_kp) == 2:
                            break

                    if len(traffic_light_kp) == 2:
                        traffic_light_x.append(int((traffic_light_kp[0].pt[0] + traffic_light_kp[1].pt[0]) / 2))
                        traffic_light_kp = []
                print('TF X: ' + str(traffic_light_x))

            window_images = None
            if len(traffic_light_x) > 0:
                windows = self.create_search_windows(traffic_light_x, img_width, img_height)
                window_images = self.get_window_images(image, windows)

            if SAVE_IMAGES:
                for marker in key_points:
                    image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 0, 255))
                misc.imsave('real_key_points.jpg', image)

                if window_images is not None:
                    for idx, img in enumerate(window_images):
                        misc.imsave('output/' + str(idx) + '.jpg', img)

            print('KP: ' + str(len(key_points)))

            prediction_idx = 0  # UNKNOWN
            if window_images is not None:
                with self.graph.as_default():
                    prediction = self.model.predict(window_images)

                    for p in prediction:
                        # TODO: Dont use hardcoded value for "3"
                        max_prob = p[0]
                        # Loop from 1 to 3
                        for idx in range(1, 4):
                            if p[idx] > max_prob:
                                max_prob = p[idx]
                                prediction_idx = idx
                        if prediction_idx > 0:
                            break

                    traffic_light_detection = self.map_detected_index_to_traffic_light(prediction_idx)

        self.printtl(prediction_idx)
        print('-------------------------------------')

        return traffic_light_detection



img_count = 0

def image_cb(msg):
    global img_count
    img_count += 1
    #if img_count % 5 != 0:
        #return

    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    classification = classifier.get_classification(img)
    print(str(classification) + ': ' + str(img_count))


classifier = TLClassifier()
#img = misc.imread('training/real/source/images/image_160.jpg')
#classifier.get_classification(img)
#exit(0)

bridge = CvBridge()
rospy.init_node('tl_detector_test')
subscriber = rospy.Subscriber('/image_color', Image, image_cb)

while True:
   rospy.spin()

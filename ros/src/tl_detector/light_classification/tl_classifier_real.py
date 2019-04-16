import os
import cv2
import math
import keras
import numpy as np
import tensorflow as tf
from scipy import misc
from styx_msgs.msg import TrafficLight


IS_TEST = True
PRINT_BOXES = True
SAVE_UNKNOWN_IMAGES = False
PUBLISH_TL_DETECTION_IMG = True

MODEL_IMG_SIZE = (64, 64)
MIN_BRIGHTNESS = 0.8
MAX_BRIGHTNESS = 1.1
DEFAULT_BRIGHTNESS = 0.9
BRIGHTNESS_STEP = 0.1
TRAFFIC_LIGHT_MIN_PROB = 0.55
MODEL_FILE_NAME = 'training/real/model_real.h5'


class TLClassifierReal(object):
    def __init__(self):
        self.brightness_factor = DEFAULT_BRIGHTNESS

        self.detector = self.create_blob_detector()
        self.kernel = np.ones((5, 5), np.float32) / 13.0

        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_path, MODEL_FILE_NAME)
        self.model = keras.models.load_model(model_path)
        self.graph = tf.get_default_graph()


    def create_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.filterByArea = True
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByCircularity = True
        params.minCircularity = 0.2
        params.minArea = 30
        params.maxArea = 1500

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


    def perform_blob_detection(self, image, processed_img):
        brightness_reduction = int(self.brightness_factor * np.sum(np.mean(processed_img)))
        processed_img = self.adjust_brightness(processed_img, -brightness_reduction)
        if IS_TEST:
            misc.imsave('real_brightness.jpg', processed_img)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
        if IS_TEST:
            misc.imsave('real_gray.jpg', processed_img)

        key_points = self.detector.detect(processed_img)

        if IS_TEST:
            image_markers = np.copy(image)
            for marker in key_points:
                image_markers = cv2.drawMarker(image_markers, tuple(int(i) for i in marker.pt), color=(0, 0, 255))
            misc.imsave('sim_key_points.jpg', image_markers)

        return key_points


    def map_detected_index_to_traffic_light(self, detected_idx):
        if detected_idx == 1:
            return TrafficLight.GREEN
        elif detected_idx == 2:
            # Yellow will also be returned as red
            return TrafficLight.RED
        elif detected_idx == 3:
            return TrafficLight.RED
        return TrafficLight.UNKNOWN


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
        y_windows = np.int(math.ceil(y_span / y_step))

        window_list = []
        left = x_center - half_window_width

        for y in range(y_windows):
            left_top = (left, np.int(y_stop - y * y_step))
            right_bottom = (x_center + half_window_width, np.int(left_top[1] + window_height))

            if left_top[1] < 0:
                dy = -left_top[1]
                left_top = (left_top[0], 0)
                right_bottom = (right_bottom[0], right_bottom[1] + dy)

            if left_top[0] >= 0 and left_top[1] >= 0 and right_bottom[0] <= img_width and right_bottom[1] <= img_height:
                window_list.append((left_top, right_bottom))

        return window_list


    def create_search_windows(self, traffic_light, img_width, img_height):
        windows = []
        overlap = (0.9, 0.9)

        for tl in traffic_light:
            if tl[1] <= 0:
                continue

            tl_width_green_red = tl[1]
            tl_height_green_red = int(tl[1] * 3.5)
            tl_width_yellow = int(tl[1] * 0.5)
            tl_height_yellow = int(tl[1] * 1.5)

            windows.append(self.slide_window(img_width, img_height, tl[0], (None, None), (tl_width_green_red, tl_height_green_red), overlap))
            windows.append(self.slide_window(img_width, img_height, tl[0], (None, None), (tl_width_yellow, tl_height_yellow), overlap))

        return windows


    def get_window_images(self, image, windows):
        new_windows = []
        window_images = []

        for window_array in windows:
            tl_images = []
            for window in window_array:
                if window[0][0] == window[1][0]:
                    continue
                window_image = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
                window_image = cv2.resize(window_image, MODEL_IMG_SIZE)
                tl_images.append(window_image)

            max_brightness = 0
            selected_img_idx = -1

            for idx, img in enumerate(tl_images):
                brightness = np.sum(cv2.mean(img))
                if brightness > max_brightness:
                    max_brightness = brightness
                    selected_img_idx = idx

            if selected_img_idx > -1:
                window_images.append(tl_images[selected_img_idx])
                new_windows.append(window_array[selected_img_idx])

        return new_windows, window_images


    def get_traffic_lights(self, key_points):
        traffic_lights = []

        for idx, kp in enumerate(key_points):
            min_y_distance = 9999
            second_kp = None

            for idx2, kp2 in enumerate(key_points):
                if idx == idx2:
                    continue
                if abs(kp.pt[0] - kp2.pt[0]) < 4:
                    distance_y = int(abs(kp.pt[1] - kp2.pt[1]))
                    if distance_y < min_y_distance:
                        min_y_distance = distance_y
                        second_kp = kp2

            if min_y_distance > 10 and second_kp is not None:
                center_x = int((kp.pt[0] + second_kp.pt[0]) / 2)
                if (center_x, min_y_distance) not in traffic_lights:
                    traffic_lights.append((center_x, min_y_distance))

        return traffic_lights


    def create_model_input_images(self, image, img_width, img_height, traffic_lights):
        if len(traffic_lights) == 0:
            return None

        windows = self.create_search_windows(traffic_lights, img_width, img_height)
        windows, window_images = self.get_window_images(image, windows)

        if IS_TEST and PRINT_BOXES:
            if window_images is not None:
                for idx, img in enumerate(window_images):
                    misc.imsave('output/' + str(idx) + '.jpg', img)

        return windows, window_images


    def print_tl_detection(self, prediction_idx):
        if prediction_idx == 1:
            print('GREEN')
        elif prediction_idx == 2:
            print('YELLOW')
        elif prediction_idx == 3:
            print('RED')
        else:
            print('UNKNOWN')


    def perform_object_detection(self, image, windows, window_images):
        with self.graph.as_default():
            prediction = self.model.predict(np.array(window_images))

        max_prob = 0
        prediction_idx = 0  # Unknown

        for window_idx, p in enumerate(prediction):
            for idx in range(1, 4):
                if p[idx] > max_prob:
                    max_prob = p[idx]
                    prediction_idx = idx
                    prediction_window_idx = window_idx

        if max_prob < TRAFFIC_LIGHT_MIN_PROB:
            prediction_idx = 0
        if prediction_idx > 0:
            if not IS_TEST and PUBLISH_TL_DETECTION_IMG:
                cv2.rectangle(image, windows[prediction_window_idx][0], windows[prediction_window_idx][1], (0, 0, 255), 2)
        else:
            self.brightness_factor -= BRIGHTNESS_STEP
            if self.brightness_factor < MIN_BRIGHTNESS:
                self.brightness_factor = MAX_BRIGHTNESS
            if not IS_TEST and SAVE_UNKNOWN_IMAGES:
                misc.imsave('unknown/' + str(self.img_idx) + '.jpg', image)
                self.img_idx += 1

        self.print_tl_detection(prediction_idx)
        traffic_light_detection = self.map_detected_index_to_traffic_light(prediction_idx)
        return traffic_light_detection


    def get_classification(self, image):
        traffic_light_detection = TrafficLight.UNKNOWN

        img_width = image.shape[1]
        img_height = int(image.shape[0] / 2)
        processed_img = image[:img_height, :]

        key_points = self.perform_blob_detection(image, processed_img)
        traffic_lights = self.get_traffic_lights(key_points)
        #print('KP: ' + str(len(key_points)))
        print(traffic_lights)

        windows, window_images = self.create_model_input_images(image, img_width, img_height, traffic_lights)

        if window_images is not None and len(window_images) > 0:
            traffic_light_detection = self.perform_object_detection(image, windows, window_images)

        if not IS_TEST and PUBLISH_TL_DETECTION_IMG:
            image = bridge.cv2_to_imgmsg(image, "rgb8")
            img_publisher.publish(image)

        return traffic_light_detection



if PUBLISH_TL_DETECTION_IMG:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    bridge = CvBridge()
    img_publisher = rospy.Publisher('/image_traffic_light', Image, queue_size=1)


#img_count = 0
def image_cb(msg):
#    global img_count
#    img_count += 1
#    if img_count % 2 != 0:
#        return

    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    classifier.get_classification(img)


classifier = TLClassifierReal()

if IS_TEST:
    img = misc.imread('training/real/source/images/image_0.jpg')
    classifier.get_classification(img)
    exit(0)

#bridge = CvBridge()
rospy.init_node('tl_detector_test')
subscriber = rospy.Subscriber('/image_color', Image, image_cb)

while True:
   rospy.spin()

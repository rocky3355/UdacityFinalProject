import os
import cv2
import math
import keras
import numpy as np
import tensorflow as tf
from scipy import misc
from styx_msgs.msg import TrafficLight


IS_TEST = False
PRINT_BOXES = False
SAVE_UNKNOWN_IMAGES = False
PUBLISH_TL_DETECTION_IMG = False

MODEL_IMG_SIZE = (64, 64)
BRIGHTNESS_FACTOR = 0.9
TRAFFIC_LIGHT_MIN_PROB = 0.5
MODEL_FILE_NAME = 'training/real/model_real.h5'


class TLClassifierReal(object):
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
        brightness_reduction = int(BRIGHTNESS_FACTOR * np.sum(np.mean(processed_img)))
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
            misc.imsave('real_key_points.jpg', image_markers)

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


    def is_window_valid(self, top_left, bottom_right, img_size):
        return top_left[0] > 0 and top_left[1] > 0 and bottom_right[0] < img_size[0] and bottom_right[1] < img_size[1]


    def calculate_patch_brightness(self, center_x, center_y, distance_y, tl_width_green_red, image, img_size):
        brightness_up = 0
        brightness_down = 0

        # Calculate upper brightness
        patch_top_left = (center_x - tl_width_green_red / 2, center_y - distance_y)
        patch_bottom_right = (patch_top_left[0] + tl_width_green_red, patch_top_left[1] + tl_width_green_red)
        if self.is_window_valid(patch_top_left, patch_bottom_right, img_size):
            brightness_up = np.sum(
                np.mean(image[patch_top_left[1]:patch_bottom_right[1], patch_top_left[0]:patch_bottom_right[0]]))

        # Calculate lower brightness
        patch_top_left = (center_x - tl_width_green_red / 2, center_y + distance_y)
        patch_bottom_right = (patch_top_left[0] + tl_width_green_red, patch_top_left[1] + tl_width_green_red)
        if self.is_window_valid(patch_top_left, patch_bottom_right, img_size):
            brightness_down = np.sum(
                np.mean(image[patch_top_left[1]:patch_bottom_right[1], patch_top_left[0]:patch_bottom_right[0]]))

        return brightness_up, brightness_down



    def create_search_windows(self, traffic_light, image, img_size):
        windows = []

        for tl in traffic_light:
            if tl[1] <= 0:
                continue

            center_x = tl[0]
            center_y = tl[1]
            distance_y = tl[2]

            tl_width_green_red = distance_y
            tl_width_yellow = int(distance_y * 0.5)

            brightness_up, brightness_down = self.calculate_patch_brightness(center_x, center_y, distance_y, tl_width_green_red, image, img_size)

            if brightness_up > brightness_down:
                # RED
                top_left = (center_x - tl_width_green_red / 2, center_y - 2 * distance_y)
                bottom_right = (top_left[0] + tl_width_green_red, int(center_y + distance_y * 1.3))
                if self.is_window_valid(top_left, bottom_right, img_size):
                    windows.append((top_left, bottom_right))

            else:
                # GREEN
                top_left = (center_x - tl_width_green_red / 2, center_y - distance_y )
                bottom_right = (top_left[0] + tl_width_green_red, int(center_y + 2.3 * distance_y))
                if self.is_window_valid(top_left, bottom_right, img_size):
                    windows.append((top_left, bottom_right))

            if distance_y > 25:
                # YELLOW
                top_left = (center_x - tl_width_yellow / 2, int(center_y - distance_y * 0.72))
                bottom_right = (top_left[0] + tl_width_yellow, int(center_y + distance_y * 0.9))
                if self.is_window_valid(top_left, bottom_right, img_size):
                    windows.append((top_left, bottom_right))

        return windows


    def get_window_images(self, image, windows):
        window_images = []

        for window in windows:
            if window[0][0] == window[1][0]:
                continue
            window_image = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            window_image = cv2.resize(window_image, MODEL_IMG_SIZE)
            window_images.append(window_image)

        return window_images


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
                center_y = int((kp.pt[1] + second_kp.pt[1]) / 2)
                if (center_x, center_y, min_y_distance) not in traffic_lights:
                    traffic_lights.append((center_x, center_y, min_y_distance))

        return traffic_lights


    def create_model_input_images(self, image, img_size, traffic_lights):
        if len(traffic_lights) == 0:
            return None

        windows = self.create_search_windows(traffic_lights, image, img_size)
        window_images = self.get_window_images(image, windows)

        if IS_TEST and PRINT_BOXES:
            if window_images is not None:
                for idx, img in enumerate(window_images):
                    misc.imsave('output/' + str(idx) + '.jpg', img)

        return windows, window_images


    def get_tl_detection_as_string(self, prediction_idx):
        if prediction_idx == 1:
            return 'Green'
        elif prediction_idx == 2:
            return 'Yellow'
        elif prediction_idx == 3:
            return 'Red'
        return 'Unknown'


    def label_img(self, image, window, prediction_idx):
        lineType = 2
        fontScale = 0.7
        fontColor = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = self.get_tl_detection_as_string(prediction_idx)
        bottomLeftCornerOfText = (window[0][0], window[0][1] - 15)

        cv2.rectangle(image, window[0], window[1], (0, 0, 255), 2)
        cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


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
                self.label_img(image, windows[prediction_window_idx], prediction_idx)
        else:
            if not IS_TEST and SAVE_UNKNOWN_IMAGES:
                misc.imsave('unknown/' + str(self.img_idx) + '.jpg', image)
                self.img_idx += 1

        #print(self.get_tl_detection_as_string(prediction_idx))

        traffic_light_detection = self.map_detected_index_to_traffic_light(prediction_idx)
        return traffic_light_detection


    def get_classification(self, image):
        traffic_light_detection = TrafficLight.UNKNOWN

        img_width = image.shape[1]
        img_height = int(image.shape[0] / 2)
        img_size = (img_width, img_height)
        processed_img = image[:img_height, :]

        key_points = self.perform_blob_detection(image, processed_img)
        traffic_lights = self.get_traffic_lights(key_points)
        #print('KP: ' + str(len(key_points)))
        #print(traffic_lights)

        windows, window_images = self.create_model_input_images(image, img_size, traffic_lights)

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



# Test code

#def image_cb(msg):
#    img = bridge.imgmsg_to_cv2(msg, "rgb8")
#    classifier.get_classification(img)


#classifier = TLClassifierReal()

#if IS_TEST:
#    img = misc.imread('training/real/source/images/image_136.jpg')
#    classifier.get_classification(img)
#    exit(0)

#rospy.init_node('tl_detector_test')
#subscriber = rospy.Subscriber('/image_color', Image, image_cb)

#while True:
#   rospy.spin()

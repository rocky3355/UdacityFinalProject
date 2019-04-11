import os
import cv2
import rospy
import math
import numpy as np
from scipy import misc
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from styx_msgs.msg import TrafficLight


MODEL_IMG_SIZE = (64, 64)
TRAFFIC_LIGHT_MIN_PROB = 0.4
MODEL_FILE_NAME = 'training/real/model_real.h5'


class TLClassifier(object):
    def __init__(self):
        self.create_blob_detector()
        self.kernel = np.ones((5, 5), np.float32)

        if IS_SIMULATION:
            self.kernel /= 13.0
        else:
            self.kernel /= 13.0

        if PERFORM_MODEL_EVALUATION:
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

        #window_images = np.array(window_images)
        return new_windows, window_images


    def create_blob_detector2(self):
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

    img_idx = 0
    brightness_factor = 0.9

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
            brightness_reduction = int(self.brightness_factor * np.sum(np.mean(processed_img)))
            processed_img = self.adjust_brightness(processed_img, -brightness_reduction)
            if IS_TEST:
                misc.imsave('real_brightness.jpg', processed_img)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            if IS_TEST:
                misc.imsave('real_gray.jpg', processed_img)
            #processed_img = 255 - processed_img
            #mask_bright = cv2.inRange(processed_img, 0, 100)
            #processed_img = cv2.bitwise_and(processed_img, mask_bright)
            #processed_img = cv2.threshold(processed_img, 0, 50, cv2.THRESH_BINARY)[1]
            #if IS_TEST:
            #    misc.imsave('real_filtered.jpg', processed_img)
            #processed_img = cv2.filter2D(processed_img, -1, self.kernel)
            #if IS_TEST:
            #    misc.imsave('real_blurred.jpg', processed_img)

            detector2 = self.create_blob_detector2()
            key_points = detector2.detect(processed_img)

            #processed_img = 255 - processed_img
            #if IS_TEST:
            #    misc.imsave('real_inverted.jpg', processed_img)
            #key_points += detector2.detect(processed_img)

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

                if second_kp is not None:
                    center_x = int((kp.pt[0] + second_kp.pt[0]) / 2)
                    if (center_x, distance_y) not in traffic_lights:
                        traffic_lights.append((center_x, min_y_distance))

            #print(traffic_lights)

            window_images = None
            if len(traffic_lights) > 0:
                windows = self.create_search_windows(traffic_lights, img_width, img_height)
                windows, window_images = self.get_window_images(image, windows)

            if IS_TEST:
                for marker in key_points:
                    image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 0, 255))
                misc.imsave('real_key_points.jpg', image)

                if IS_TEST and PRINT_BOXES:
                    if window_images is not None:
                        for idx, img in enumerate(window_images):
                            misc.imsave('output/' + str(idx) + '.jpg', img)

            #print('KP: ' + str(len(key_points)))

            if PERFORM_MODEL_EVALUATION:
                prediction_idx = 0  # UNKNOWN

                if window_images is not None and len(window_images) > 0:
                    with self.graph.as_default():
                        tl_predictions = []
                        max_prob = 0
                        prediction_window_idx = -1
                        #prediction_tl_idx = -1

                        #for tl_idx, tl_images in enumerate(window_images):
                        #    if len(tl_images) == 0:
                        #        continue

                        prediction = self.model.predict(np.array(window_images))

                        #probs = np.zeros(3)

                        for window_idx, p in enumerate(prediction):
                            # TODO: Dont use hardcoded value for "3"
                            # Loop from 1 to 3

                            for idx in range(1, 4):
                                #probs[idx-1] += p[idx]
                                if p[idx] > max_prob:
                                    max_prob = p[idx]
                                    prediction_idx = idx
                                    #prediction_tl_idx = tl_idx
                                    prediction_window_idx = window_idx
                                #window_idx += 1

                        #tl_predictions.append(probs)

                        if max_prob < TRAFFIC_LIGHT_MIN_PROB:
                            prediction_idx = 0
                        if prediction_idx > 0:
                            #self.brightness_factor = 0.9
                            cv2.rectangle(image, windows[prediction_window_idx][0], windows[prediction_window_idx][1], (0, 0, 255), 2)
                        else:
                            self.brightness_factor -= 0.1
                            if self.brightness_factor == 0.8:
                                self.brightness_factor = 1.1
                            if not IS_TEST and SAVE_UNKNOWN_IMAGES:
                                misc.imsave('unknown/' + str(self.img_idx) + '.jpg', image)
                                self.img_idx += 1

                        print(self.brightness_factor)
                        traffic_light_detection = self.map_detected_index_to_traffic_light(prediction_idx)

                self.printtl(prediction_idx)
                if not IS_TEST:
                    image = bridge.cv2_to_imgmsg(image, "rgb8")
                    img_publisher.publish(image)
        #print('-------------------------------------')

        return traffic_light_detection



IS_TEST = False
PRINT_BOXES = True
IS_SIMULATION = False
SAVE_UNKNOWN_IMAGES = False
PERFORM_MODEL_EVALUATION = True


if PERFORM_MODEL_EVALUATION:
    import keras
    import tensorflow as tf

img_count = 0

def image_cb(msg):
    global img_count
    img_count += 1
    if img_count % 2 != 0:
        return

    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    classification = classifier.get_classification(img)
    #print(str(classification) + ': ' + str(img_count))



classifier = TLClassifier()

if IS_TEST:
    img = misc.imread('test/test30.jpg')
    classifier.get_classification(img)
    exit(0)

bridge = CvBridge()
rospy.init_node('tl_detector_test')
subscriber = rospy.Subscriber('/image_color', Image, image_cb)
img_publisher = rospy.Publisher('/image_traffic_light', Image, queue_size=1)

while True:
   rospy.spin()

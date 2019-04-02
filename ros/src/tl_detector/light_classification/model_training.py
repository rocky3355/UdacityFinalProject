import cv2
import keras
import rospy
from keras.utils import to_categorical
import tensorflow as tf
from scipy import misc
from keras.layers import *
from keras.models import Sequential
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


NUMBER_OF_CLASSES = 4
MODEL_IMG_SIZE = (64, 64)
MODEL_FILE_NAME = 'training/simulation/model_simulation.h5'
TRAIN_DIR = 'training/simulation/processed/'
LABEL_TEXT_FILE = TRAIN_DIR + '/labels.txt'
TRAFFIC_LIGHTS = ['Unknown', 'Green', 'Yellow', 'Red']


def slide_window(img_shape, x_start_stop, y_start_stop, xy_window, xy_overlap):
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


def create_search_windows(img_shape):
    overlap = (0.5, 0.5)
    near_windows = slide_window(img_shape, (100, 700), (0, 200), (50, 160), overlap)
    mid_windows = slide_window(img_shape, (200, 600), (0, 200), (40, 120), overlap)
    far_windows = slide_window(img_shape, (300, 500), (0, 200), (20, 60), overlap)
    windows = near_windows + mid_windows + far_windows
    return windows


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(NUMBER_OF_CLASSES))
    #model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='mse')
    return model


def load_model():
    model = keras.models.load_model(MODEL_FILE_NAME)
    return model


def train_model(model):
    images = []
    labels = []
    img_files = []
    label_lines = None

    with open(LABEL_TEXT_FILE) as file:
        label_lines = file.readlines()

    for line in label_lines:
        line_parts = line.split(';')
        labels.append(int(line_parts[0]))
        img_files.append(line_parts[1].strip())

    for img_file in img_files:
        img = misc.imread(img_file)  # [:, :, :3]
        images.append(img)

    images = np.array(images)
    labels = np.array(to_categorical(labels, num_classes=NUMBER_OF_CLASSES))

    model.fit(images, labels, epochs=20, validation_split=0.2, shuffle=True)
    model.save(MODEL_FILE_NAME)


def get_window_images(img):
    global windows

    #idx = 0
    window_imgs = []
    for window in windows:
        window_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        window_img = cv2.resize(window_img, MODEL_IMG_SIZE)
        #misc.imsave('boxes/' + str(idx) + '.jpg', window_img)
        #idx += 1
        window_imgs.append(window_img)
        #cv2.rectangle(img, window[0], window[1], (0, 0, 255), 6)

    #misc.imsave('test.jpg', img)
    #exit(0)
    window_imgs = np.array(window_imgs)
    return window_imgs

img_count = 0

def image_cb(msg):
    global graph, img_count

    img_count += 1
    if img_count % 5 != 0:
        return

    img = bridge.imgmsg_to_cv2(msg, "rgb8")
    window_images = get_window_images(img)

    with graph.as_default():
        result = model.predict(window_images)

    traffic_light = None
    #print(result)

   # max_prob = 0

    for window in result:
        for idx in range(1, NUMBER_OF_CLASSES):
            prob = window[idx]
            if prob > 0.6:
                traffic_light = TRAFFIC_LIGHTS[idx]
                break
        if traffic_light is not None:
            break

    if traffic_light is None:
        traffic_light = TRAFFIC_LIGHTS[0]

    print(traffic_light)


TRAIN_MODEL = True


if TRAIN_MODEL:
    model = create_model()
    train_model(model)
    exit(0)

model = load_model()
graph = tf.get_default_graph()

bridge = CvBridge()
img_shape = (800, 600)
windows = create_search_windows(img_shape)

rospy.init_node('tl_training')
subscriber = rospy.Subscriber('/image_color', Image, image_cb)

while True:
   rospy.spin()

img = misc.imread('test/test9.jpg')[:,:,:3]
img = cv2.resize(img, MODEL_IMG_SIZE)
img = np.array([img])
result = model.predict(img)
print(result)

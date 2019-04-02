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

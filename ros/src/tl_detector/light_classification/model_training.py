import cv2
import keras
from keras.utils import to_categorical
import tensorflow as tf
from scipy import misc
from keras.layers import *
from keras.models import Sequential


NUMBER_OF_CLASSES = 4
MODEL_IMG_SIZE = (64, 64)
TRAIN_DIR = 'training/real/processed/'
LABEL_TEXT_FILE = TRAIN_DIR + '/labels.txt'
MODEL_FILE_NAME = 'training/real/model_real.h5'
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
    model.compile(optimizer='adam', loss='mse')
    return model


def load_model():
    model = keras.models.load_model(MODEL_FILE_NAME)
    return model


def train_model(model):
    images = []
    labels = []
    img_files = []

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


TRAIN_MODEL = False

if TRAIN_MODEL:
    model = create_model()
    train_model(model)
    exit(0)

model = load_model()
graph = tf.get_default_graph()

img = misc.imread('output/1.jpg')
img = cv2.resize(img, MODEL_IMG_SIZE)
img = np.array([img])
result = model.predict(img)
print(result)

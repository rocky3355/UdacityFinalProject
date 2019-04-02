import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

saved_img_counter = 0
total_img_counter = 0
bridge = CvBridge()


def image_cb(img):
    global saved_img_counter, total_img_counter
    total_img_counter += 1
    if total_img_counter % 10 != 0:
        return

    cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    cv2.imwrite('/home/qxw0266/Udacity/UdacityFinalProject/ros/src/tl_detector/light_classification/training/simulation/source/image_' + str(saved_img_counter) + '.jpg', cv_image)
    saved_img_counter += 1


rospy.init_node('image_recorder')
subscriber = rospy.Subscriber('/image_color', Image, image_cb)

while True:
    rospy.spin()
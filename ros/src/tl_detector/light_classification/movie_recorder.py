import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def image_cb(msg):
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    video.write(img)


video = cv2.VideoWriter('traffic_light.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (800, 600))
bridge = CvBridge()
rospy.init_node('movie_recorder')
subscriber = rospy.Subscriber('/image_traffic_light', Image, image_cb)

while True:
   rospy.spin()
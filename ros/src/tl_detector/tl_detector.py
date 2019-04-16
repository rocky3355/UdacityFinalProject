#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier_real import TLClassifierReal
from light_classification.tl_classifier_simulation import TLClassifierSimulation
import tf
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        config_file = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_file)

        self.stop_lines_wp_idx = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        if self.config['is_site']:
            global STATE_COUNT_THRESHOLD
            STATE_COUNT_THRESHOLD = 5
            self.light_classifier = TLClassifierReal()
        else:
            self.light_classifier = TLClassifierSimulation()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        for stop_line in self.config['stop_line_positions']:
            stop_line_pose = PoseStamped()
            stop_line_pose.pose.position.x = stop_line[0]
            stop_line_pose.pose.position.y = stop_line[1]
            stop_line_pose.pose.position.z = 0
            stop_line_wp_idx = self.get_closest_waypoint(stop_line_pose)
            self.stop_lines_wp_idx.append(stop_line_wp_idx)
            #print("Adding stop line for #" + str(stop_line_wp_idx))


    def traffic_cb(self, msg):
        self.lights = msg.lights


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        if self.waypoints is None:
            return

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def distance_squared(self, pose1, pose2):
        pos1 = pose1.pose.position
        pos2 = pose2.pose.position
        delta_x = pos1.x - pos2.x
        delta_y = pos1.y - pos2.y
        delta_z = pos1.z - pos2.z
        dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        return dist

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        prev_distance = sys.float_info.max
        closest_wp_idx = -1

        for idx, wp in enumerate(self.waypoints.waypoints):
            distance = self.distance_squared(wp.pose, pose)
            if distance < prev_distance:
                prev_distance = distance
                closest_wp_idx = idx
            else:
                break

        return closest_wp_idx


    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def get_stop_line_wp_idx(self, light_wp_idx):
        min_idx_diff = 9999999
        closest_stop_line_wp_idx = -1

        for stop_line_wp_idx in self.stop_lines_wp_idx:
            idx_diff = abs(stop_line_wp_idx - light_wp_idx)
            if idx_diff < min_idx_diff:
                min_idx_diff = idx_diff
                closest_stop_line_wp_idx = stop_line_wp_idx

        return closest_stop_line_wp_idx


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # List of positions that correspond to the line to stop in front of for a given intersection
        #stop_line_positions = self.config['stop_line_positions']

        min_idx_diff = 99999
        closest_stop_line_wp_idx = -1

        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose)
            #print(car_wp_idx)
            for tr_light in self.lights:
                light_wp_idx = self.get_closest_waypoint(tr_light.pose)
                stop_line_wp_idx = self.get_stop_line_wp_idx(light_wp_idx)
                #print(str(light_wp_idx) + "  /  " + str(stop_line_wp_idx))
                # TODO: What if car is near last waypoint and traffic near first waypoints?
                idx_diff = abs(stop_line_wp_idx - car_wp_idx)
                if idx_diff < min_idx_diff:
                    min_idx_diff = idx_diff
                    closest_stop_line_wp_idx = stop_line_wp_idx

        if closest_stop_line_wp_idx > -1:
            state = self.get_light_state()
            #print('State: ' + str(state) + "  /  Stopline: #" + str(closest_stop_line_wp_idx) + "  /  Car: #" + str(car_wp_idx))
            return closest_stop_line_wp_idx, state

        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

#!/usr/bin/env python

import sys
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50

class WaypointUpdater(object):
    def __init__(self):
        self.ego_pose = PoseStamped()

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.velocity = 0
        self.num_waypoints = 0
        self.waypoints = None
        self.original_velocities = None
        self.update_wp_velocities = True
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        rospy.spin()


    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        self.num_waypoints = len(self.waypoints)

        self.original_velocities = np.zeros(self.num_waypoints)
        for wp_idx in range(self.num_waypoints):
            self.original_velocities[wp_idx] = self.waypoints[wp_idx].twist.twist.linear.x


    def velocity_cb(self, msg):
        self.velocity = msg.twist.linear.x


    def pose_cb(self, pose):
        if self.waypoints is None:
            return

        lane = Lane()

        prev_distance = sys.float_info.max
        start_idx = -1

        for idx,wp in enumerate(self.waypoints):
            distance = self.distance_squared(wp.pose, pose)
            if distance < prev_distance:
                prev_distance = distance
                start_idx = idx
            else:
                break

        #print("Pose: " + str(start_idx))

        for i in range(LOOKAHEAD_WPS):
            idx = (i + start_idx)
            if idx >= self.num_waypoints:
                break

            if self.update_wp_velocities:
                velocity = self.original_velocities[idx]
                self.set_waypoint_velocity(idx, velocity)
            lane.waypoints.append(self.waypoints[idx])

        self.final_waypoints_pub.publish(lane)


    def traffic_cb(self, msg):
        light_wp_idx = msg.data
        if light_wp_idx < 0 or self.waypoints is None:
            self.update_wp_velocities = True
            return

        if not self.update_wp_velocities:
            return

        #print("Red light at: #" + str(light_wp_idx))
        self.update_wp_velocities = False

        standstill_offset = 2
        standstill_wp_idx = light_wp_idx - standstill_offset

        for i in range(standstill_offset + 1):
            wp_idx = light_wp_idx - i
            self.set_waypoint_velocity(wp_idx, 0.0)
            #print('Set #' + str(wp_idx) + ' to 0.0')

        velocity = self.velocity

        if velocity < 5.0:
            velocity = 5.0

        steps = int(velocity * 4.0)
        #print('Steps: ' + str(steps) + '  /  Velocity: ' + str(self.velocity))
        vel_step = velocity / steps

        for i in range(steps):
            wp_idx = (standstill_wp_idx - i - 1) % self.num_waypoints
            wp_velocity = i * vel_step
            self.set_waypoint_velocity(wp_idx, wp_velocity)
            #print('Set #' + str(wp_idx) + ' to ' + str(wp_velocity))


    def obstacle_cb(self, msg):
        pass


    def get_waypoint_velocity(self, idx):
        return self.waypoints[idx].twist.twist.linear.x

    def set_waypoint_velocity(self, idx, velocity):
        self.waypoints[idx].twist.twist.linear.x = velocity


    def distance_squared(self, pose1, pose2):
        pos1 = pose1.pose.position
        pos2 = pose2.pose.position
        delta_x = pos1.x - pos2.x
        delta_y = pos1.y - pos2.y
        delta_z = pos1.z - pos2.z
        dist = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

import rospy
from pid import PID
from yaw_controller import YawController


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, yaw_controller):
        self.throttle_pid = PID(0.1, 0.02, 0, 0.0, 1.0)
        # TODO: Use same pid for throttle and brake, limit by maxdecel/maxaccel
        self.brake_pid = PID(200.0, 0.0, 0.0, 0.0, 1000.0)
        #self.steer_pid = PID(5.0, 1.0, 0.0, -8.0, 8.0)
        self.last_msg_timestamp = None
        self.yaw_controller = yaw_controller


    def control(self, current_velocity, target_velocity):
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        msg_time = rospy.get_rostime()

        if self.last_msg_timestamp is None:
            self.last_msg_timestamp = msg_time
            return throttle, brake, steer

        delta_time = msg_time - self.last_msg_timestamp
        delta_time_sec = delta_time.secs + 1.0E-9 * delta_time.nsecs
        if delta_time_sec == 0:
            return throttle, brake, steer

        self.last_msg_timestamp = msg_time

        vel_diff = target_velocity.twist.linear.x - current_velocity.twist.linear.x

        # TODO: Limit values
        # TODO: Reset when dbw_enabled toggles (also is_first_message)
        if current_velocity.twist.linear.x < 0.5 and target_velocity.twist.linear.x < 0.1:
            brake = 1000
        elif vel_diff > 0:
            throttle = self.throttle_pid.step(vel_diff, delta_time_sec)
        elif vel_diff < -0.5:
            brake = self.brake_pid.step(-vel_diff, delta_time_sec)

        steer = 2.0 * self.yaw_controller.get_steering(target_velocity.twist.linear.x, target_velocity.twist.angular.z, current_velocity.twist.linear.x)

        #print(str(current_velocity.twist.linear.x) + ' / ' + str(target_velocity.twist.linear.x) + '   --   ' + str(target_velocity.twist.angular.z) + '  /  ' + str(steer))
        print(str(vel_diff) + " / " + str(throttle) + " / " + str(brake))

        return throttle, brake, steer

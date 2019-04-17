import rospy
from pid import PID


GAS_DENSITY = 2.858
ONE_MPH = 0.44704
STANDSTILL_BREAK_TORQUE = 1000


class Controller(object):
    def __init__(self, yaw_controller):
        self.throttle_pid = PID(0.04, 0.01, 0.0, 0.0, 0.4)
        self.brake_pid = PID(150.0, 0.0, 0.0, 0.0, 500.0)
        self.last_msg_timestamp = None
        self.yaw_controller = yaw_controller


    def reset(self):
        self.throttle_pid.reset()
        self.brake_pid.reset()
        self.last_msg_timestamp = None


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

        if current_velocity.twist.linear.x < 1.0:
            steer = 0
        else:
            steer = self.yaw_controller.get_steering(target_velocity.twist.linear.x, target_velocity.twist.angular.z, current_velocity.twist.linear.x)

        if current_velocity.twist.linear.x < 0.5 and target_velocity.twist.linear.x < 0.1:
            brake = STANDSTILL_BREAK_TORQUE
        elif vel_diff > 0:
            throttle = self.throttle_pid.step(vel_diff, delta_time_sec)
        elif vel_diff < -0.5:
            brake = self.brake_pid.step(-vel_diff, delta_time_sec)

        return throttle, brake, steer

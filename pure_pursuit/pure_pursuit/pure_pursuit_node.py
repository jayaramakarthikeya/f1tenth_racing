#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # TODO: create ROS subscribers and publishers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_sub_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/waypoint', 10)

        # parameters
        velocity = 6.0
        waypoint = np.array([
            [1.0, 0.2], 
            [0.46, 22.76], 
            [-8.36, 22.48], 
            [-7.63, 0.07], 
            [1.0, 0.2]])
        lookahead_distance = 2.0
        lookahead_time = lookahead_distance/velocity
        time_resolution = 0.025
        self.steer_gain = 1.0
        self.speed_gain = 1.0/lookahead_time
        self.max_steering_angle = 0.36

        distance = np.diff(waypoint, axis=0)
        distance = np.linalg.norm(distance, axis=1)
        time = np.cumsum(distance/velocity)
        time = np.insert(time, 0, 0)
        linear_interp = interp1d(time, waypoint, kind='linear', axis=0)

        trajectory_time = np.arange(0, time[-1], time_resolution)
        trajectory_position = linear_interp(trajectory_time)

        shift = int(lookahead_time/time_resolution)
        lookahead_trajectory_time = np.roll(trajectory_time, -shift)
        lookahead_trajectory_position = linear_interp(lookahead_trajectory_time)

        self.kn_regressor = KNeighborsRegressor()
        self.kn_regressor.fit(trajectory_position, lookahead_trajectory_position)

        print("PurePursuit Initialized")


    def odom_sub_callback(self, odom_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        current_position = np.array([[odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]])
        lookahead_position = self.kn_regressor.predict(current_position)

        # TODO: transform goal point to vehicle frame of reference
        arb = R.from_quat([
            odom_msg.pose.pose.orientation.x, 
            odom_msg.pose.pose.orientation.y, 
            odom_msg.pose.pose.orientation.z, 
            odom_msg.pose.pose.orientation.w])
        lookahead_position_in_map_frame = np.insert(lookahead_position-current_position, 2, 0)
        lookahead_position_in_vehicle_frame = arb.inv().apply(lookahead_position_in_map_frame)

        # TODO: calculate curvature/steering angle and speed
        L = np.linalg.norm(lookahead_position_in_vehicle_frame)
        gamma = lookahead_position_in_vehicle_frame[1]/(L**2)
        steering_angle = self.steer_gain * gamma
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        speed = self.speed_gain * L

        # TODO: publish drive message, don't forget to limit the steering angle.
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.get_logger().info('Publishing: steering_angle: %f, speed: %f' % (steering_angle, speed))
        self.drive_pub.publish(drive_msg)

        # publish the waypoint
        waypoint_msg = PointStamped()
        waypoint_msg.point.x = lookahead_position[0][0]
        waypoint_msg.point.y = lookahead_position[0][1]
        waypoint_msg.header.frame_id = "map"
        self.waypoint_pub.publish(waypoint_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

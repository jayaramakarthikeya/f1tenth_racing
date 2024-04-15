#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
import tf2_ros
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
import transforms3d
from nav_msgs.msg import Odometry
import math

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypointlogger_node')
        
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.save_waypoint, 10)
        self.file = open('waypoints'+'.csv', 'w')

    def save_waypoint(self,data):
        quaternion = np.array([data.pose.pose.orientation.x, 
                            data.pose.pose.orientation.y, 
                            data.pose.pose.orientation.z, 
                            data.pose.pose.orientation.w])

        euler = transforms3d.euler.quat2euler(quaternion)
        speed = LA.norm(np.array([data.twist.twist.linear.x, 
                                data.twist.twist.linear.y, 
                                data.twist.twist.linear.z]),2)
        
        ori = data.pose.pose.orientation
        if np.abs(speed)>0.0:
            print(euler)
            print(np.arctan2(2.0 * (ori.w * ori.z + ori.x * ori.y), 1 - 2.0 * (ori.y**2 + ori.z**2)))

            self.file.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                            data.pose.pose.position.y,
                                            euler[0],
                                            speed))

def main(args=None):
    rclpy.init(args=args)
    waypoint_logger_node = WaypointLogger()
    rclpy.spin(waypoint_logger_node)

    waypoint_logger_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_dilation
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import interp1d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker

# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
class Vertex:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

    def __repr__(self) -> str:
        return f"Vertex({self.x}, {self.y})"

# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"
        map_topic = "/map_server/map"
        path_topic = "/rrt_path"
        tree_topic = "/rrt_tree"
        goal_topic = "/rrt_goal"
        lookahead_topic = "/rrt_lookahead"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        
        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.drive_pub_ = self.create_publisher(
            AckermannDriveStamped,
            "drive",
            1)
        
        self.path_pub_ = self.create_publisher(
            Marker,
            path_topic,
            1)
        
        self.tree_pub_ = self.create_publisher(
            Marker,
            tree_topic,
            1)
        
        self.goal_pub_ = self.create_publisher(
            PointStamped,
            goal_topic,
            1)
        
        self.lookahead_pub_ = self.create_publisher(
            PointStamped,
            lookahead_topic,
            1)

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.map_client = self.create_client(GetMap, map_topic)
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        request = GetMap.Request()
        future = self.map_client.call_async(request)
        future.add_done_callback(self.map_callback)

        self.curr_position = None
        self.curr_orientation = None

        # flags
        self.map_init = True
        self.pose_init = True
        self.scan_init = True

        # constants
        self.margin = 0.2
        self.margin_index = None
        self.iter_max = 100
        self.step_len = 50.0
        self.goal_sample_rate = 0.1
        self.search_radius = 100.0
        self.rng = np.random.default_rng()

        self.lookahead_distance = 0.5
        self.velocity = 3.0
        self.steering_gain = 1.0
        self.max_steering_angle = 0.36

        # pure pursuit
        # parameters
        velocity = 6.0
        waypoint = np.array([
            [1.0, 0.2], 
            [0.46, 22.76], 
            [-8.36, 22.48], 
            [-7.63, 0.07], 
            [1.0, 0.2]])
        lookahead_distance = 5.0
        lookahead_time = lookahead_distance/velocity
        time_resolution = 0.025

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

    def map_callback(self, future):
        response = future.result()

        # Set the origin of the occupancy grid
        self.r_map_world = np.array(
            [response.map.info.origin.position.x, 
                response.map.info.origin.position.y])
        
        # Set the resolution of the occupancy grid
        self.resolution = response.map.info.resolution

        # Set margin index
        self.margin_index = int(self.margin / self.resolution)

        # Set the flag to indicate that the occupancy grid has been received
        self.map_init = False

        # Convert the occupancy grid to a numpy array
        self.static_map = np.array(response.map.data).reshape(
            (response.map.info.height, response.map.info.width)).astype(bool)
        binary_dilation(
            self.static_map, 
            iterations=self.margin_index, 
            output=self.static_map)
        
        # Preallocate the dynamic map and zero map
        self.dynamic_map = np.empty_like(self.static_map)
        self.zero_map = np.zeros_like(self.static_map)

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # status check
        if self.map_init:
            return
        if self.pose_init:
            return
        if self.scan_init:
            theta = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
            self.cos_theta = np.cos(theta)
            self.sin_theta = np.sin(theta)
            self.zero_ranges = np.zeros(len(scan_msg.ranges))
            self.scan_init = False
        
        # update occupancy grid
        obs_in_lidar_frame = np.vstack([
            scan_msg.ranges * self.cos_theta, 
            scan_msg.ranges * self.sin_theta,
            self.zero_ranges])
        obs_in_world_frame = self.curr_position + self.curr_orientation.apply(obs_in_lidar_frame.T)
        obs_in_map_frame = np.round((obs_in_world_frame[:, :2] - self.r_map_world) / self.resolution).astype(int)
        obs_in_map_frame = obs_in_map_frame[(obs_in_map_frame[:, 0] >= 0) & (obs_in_map_frame[:, 0] < self.static_map.shape[1])]
        obs_in_map_frame = obs_in_map_frame[(obs_in_map_frame[:, 1] >= 0) & (obs_in_map_frame[:, 1] < self.static_map.shape[0])]
        np.copyto(self.dynamic_map, self.zero_map)
        self.dynamic_map[obs_in_map_frame[:, 1], obs_in_map_frame[:, 0]] = 1
        binary_dilation(
            self.dynamic_map, 
            iterations=self.margin_index, 
            output=self.dynamic_map)
        np.logical_or(self.static_map, self.dynamic_map, out=self.dynamic_map)

        # # visualize the occupancy grid
        # marker = Marker()
        # marker.header.frame_id = "map"
        # marker.header.stamp = self.get_clock().now().to_msg()
        # marker.ns = "occupancy_grid"
        # marker.id = 0
        # marker.type = Marker.CUBE_LIST
        # marker.action = Marker.ADD
        # marker.pose.orientation.w = 1.0
        # marker.scale.x = self.resolution
        # marker.scale.y = self.resolution
        # marker.scale.z = 0.1
        # marker.color.a = 1.0
        # for i in range(self.dynamic_map.shape[0]):
        #     for j in range(self.dynamic_map.shape[1]):
        #         if self.dynamic_map[i, j]:
        #             point = Point()
        #             point.x = j * self.resolution + self.r_map_world[0]
        #             point.y = i * self.resolution + self.r_map_world[1]
        #             marker.points.append(point)
        # self.marker_pub_.publish(marker)

        # get goal position
        goal_position = self.kn_regressor.predict(np.array([self.curr_position[:2]]))
        goal_position = goal_position[0]

        # publish goal point
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "map"
        point_msg.point.x = goal_position[0]
        point_msg.point.y = goal_position[1]
        self.goal_pub_.publish(point_msg)

        # transform current and goal position to map frame
        curr_position_map_frame = np.round((self.curr_position[:2] - self.r_map_world) / self.resolution).astype(int)
        goal_position_map_frame = np.round((goal_position - self.r_map_world) / self.resolution).astype(int)

        # get the bounds for sampling
        radius = max(
            abs(goal_position_map_frame[0] - curr_position_map_frame[0]), 
            abs(goal_position_map_frame[1] - curr_position_map_frame[1]))
        sample_x_min = min(curr_position_map_frame[0], goal_position_map_frame[0]) - radius
        sample_x_max = max(curr_position_map_frame[0], goal_position_map_frame[0]) + radius
        sample_y_min = min(curr_position_map_frame[1], goal_position_map_frame[1]) - radius
        sample_y_max = max(curr_position_map_frame[1], goal_position_map_frame[1]) + radius

        # RRT star algorithm
        curr_vertex = Vertex(curr_position_map_frame[0], curr_position_map_frame[1])
        vertices = [curr_vertex]
        for _ in range(self.iter_max):
            sample_vertex = self.sample(sample_x_min, sample_x_max, sample_y_min, sample_y_max, goal_position_map_frame)
            nearest_vertex = self.nearest(vertices, sample_vertex)
            new_vertex = self.steer(nearest_vertex, sample_vertex)
            if self.check_collision(nearest_vertex, new_vertex):
                neighborhood_indices = self.near(vertices, new_vertex)
                if neighborhood_indices:
                    new_vertex = self.choose_parent(vertices, neighborhood_indices, new_vertex)
                    self.rewire(vertices, neighborhood_indices, new_vertex)
                vertices.append(new_vertex)
                # if self.is_goal(new_vertex, goal_position_map_frame[0], goal_position_map_frame[1]):
                #     break
        path = self.find_path(vertices, goal_position_map_frame[0], goal_position_map_frame[1])
        
        # visualize the path
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.a = 1.0
        for vertex in path:
            point = Point()
            point.x = vertex.x * self.resolution + self.r_map_world[0]
            point.y = vertex.y * self.resolution + self.r_map_world[1]
            marker.points.append(point)
        self.path_pub_.publish(marker)

        # visualize the tree
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rrt_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.color.g = 1.0
        marker.color.a = 1.0
        for vertex in vertices:
            if vertex.parent:
                point = Point()
                point.x = vertex.x * self.resolution + self.r_map_world[0]
                point.y = vertex.y * self.resolution + self.r_map_world[1]
                marker.points.append(point)
                point = Point()
                point.x = vertex.parent.x * self.resolution + self.r_map_world[0]
                point.y = vertex.parent.y * self.resolution + self.r_map_world[1]
                marker.points.append(point)
        self.tree_pub_.publish(marker)

        # pure pursuit
        for vertex in path:
            if self.get_distance(vertex, curr_vertex) > self.lookahead_distance:
                break
        lookahead_position = np.array([
            vertex.x * self.resolution + self.r_map_world[0], 
            vertex.y * self.resolution + self.r_map_world[1],
            0.0])
        lookahead_position_vehicle_frame = self.curr_orientation.inv().apply(lookahead_position - self.curr_position)[:2]
        L = np.linalg.norm(lookahead_position_vehicle_frame)
        gamma = lookahead_position_vehicle_frame[1] / (L**2)
        steering_angle = self.steering_gain * gamma
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.velocity
        self.drive_pub_.publish(drive_msg)

        # publish lookahead point
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "map"
        point_msg.point.x = lookahead_position[0]
        point_msg.point.y = lookahead_position[1]
        self.lookahead_pub_.publish(point_msg)
        
    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        if self.pose_init:
            self.pose_init = False
        self.curr_position = np.array([
            pose_msg.pose.pose.position.x, 
            pose_msg.pose.pose.position.y,
            pose_msg.pose.pose.position.z])
        self.curr_orientation = R.from_quat([
            pose_msg.pose.pose.orientation.x,
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
            pose_msg.pose.pose.orientation.w])

    def sample(self, sample_x_min, sample_x_max, sample_y_min, sample_y_max, goal_position_map_frame):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        if self.rng.uniform(0, 1) > self.goal_sample_rate:
            x = self.rng.integers(sample_x_min, sample_x_max)
            y = self.rng.integers(sample_y_min, sample_y_max)
        else:
            x = goal_position_map_frame[0]
            y = goal_position_map_frame[1]
        return Vertex(x, y)

    @staticmethod
    def nearest(vertices, sample_vertex):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        return vertices[int(np.argmin(
            [math.hypot(
                vertex.x-sample_vertex.x, 
                vertex.y-sample_vertex.y) for vertex in vertices]))]
    
    @staticmethod
    def get_distance(vertex1, vertex2):
        return math.hypot(vertex1.x-vertex2.x, vertex1.y-vertex2.y)
    
    @staticmethod
    def get_angle(vertex1, vertex2):
        return math.atan2(vertex2.y-vertex1.y, vertex2.x-vertex1.x)

    def steer(self, nearest_vertex, sample_vertex):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        dist = self.get_distance(nearest_vertex, sample_vertex)
        angle = self.get_angle(nearest_vertex, sample_vertex)
        dist = min(self.step_len, dist)
        x = round(nearest_vertex.x + dist * math.cos(angle))
        y = round(nearest_vertex.y + dist * math.sin(angle))
        parent = nearest_vertex
        return Vertex(x, y, parent)

    def check_collision(self, nearest_vertex, new_vertex):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        # Bresenham's line algorithm
        x0, y0 = nearest_vertex.x, nearest_vertex.y
        x1, y1 = new_vertex.x, new_vertex.y
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if y0 < 0 or y0 >= self.dynamic_map.shape[0] or x0 < 0 or x0 >= self.dynamic_map.shape[1]:
                return False
            if self.dynamic_map[y0, x0]:
                return False
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        return self.get_distance(latest_added_node, Vertex(goal_x, goal_y)) < self.margin_index


    def find_path(self, vertices, goal_x, goal_y):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        vertex = vertices[np.argmin(
            [self.get_distance(vertex, Vertex(goal_x, goal_y)) for vertex in vertices])]
        path = [vertex]
        while vertex.parent:
            path.append(vertex.parent)
            vertex = vertex.parent
        return path[::-1]



    # The following methods are needed for RRT* and not RRT
    def near(self, vertices, new_vertex):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        n = len(vertices) + 1
        r = min(self.search_radius * math.sqrt(math.log(n)/n), self.step_len)
        neighborhood_indices = [i for i, vertex in enumerate(vertices) if self.get_distance(vertex, new_vertex) < r]
        return neighborhood_indices
    
    def cost(self, vertex):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        cost = 0.0
        while vertex.parent:
            cost += self.get_distance(vertex, vertex.parent)
            vertex = vertex.parent
        return cost
    
    def get_new_cost(self, vertex_from, vertex_to):
        return self.cost(vertex_from) + self.get_distance(vertex_from, vertex_to)

    def choose_parent(self, vertices, neighborhood_indices, new_vertex):
        """
        This method should return the best parent in the neighborhood of the given node

        Args:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
            node (Node): current node we're finding a parent for
        Returns:
            best_parent (Node): best parent node in the neighborhood
        """
        best_parent = new_vertex.parent
        best_cost = self.cost(new_vertex)
        for index in neighborhood_indices:
            if self.get_new_cost(vertices[index], new_vertex) < best_cost and self.check_collision(vertices[index], new_vertex):
                best_parent = vertices[index]
                best_cost = self.get_new_cost(vertices[index], new_vertex)
        new_vertex.parent = best_parent
        return new_vertex
    
    def rewire(self, vertices, neighborhood_indices, new_vertex):
        """
        This method should rewire the tree given the neighborhood and the new node

        Args:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
            new_node (Node): new node that needs to be rewired
        Returns:
        """
        for index in neighborhood_indices:
            if self.get_new_cost(new_vertex, vertices[index]) < self.cost(vertices[index]) and self.check_collision(new_vertex, vertices[index]):
                vertices[index].parent = new_vertex

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
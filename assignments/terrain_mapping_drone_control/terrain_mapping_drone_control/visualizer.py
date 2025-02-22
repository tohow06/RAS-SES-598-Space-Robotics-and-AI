#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np


class PoseVisualizer(Node):
    """
    A ROS2 node that visualizes the drone's pose, path, and velocity.
    """

    def __init__(self):
        super().__init__('pose_visualizer')

        # Configure QoS profile for PX4 communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Initialize vehicle state variables
        self.vehicle_odometry = VehicleOdometry()
        self.setpoint_position = np.array([0.0, 0.0, 0.0])
        self.last_odom_update = 0.0

        # Path visualization parameters
        self.trail_size = 1000
        self.vehicle_path_msg = Path()
        self.vehicle_path_msg.header.frame_id = "map"
        self.setpoint_path_msg = Path()
        self.setpoint_path_msg.header.frame_id = "map"

        # Create subscribers
        self._setup_subscribers(qos_profile)

        # Create publishers
        self._setup_publishers()

        # Create timer for visualization updates
        self.create_timer(0.05, self.cmdloop_callback)  # 20Hz update rate

        # Parameters
        self.declare_parameter("path_clearing_timeout", -1.0)

        self.get_logger().info('Pose visualizer node initialized')

    def _setup_subscribers(self, qos_profile):
        """Set up subscriber nodes."""
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile
        )
        self.setpoint_sub = self.create_subscription(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            self.trajectory_setpoint_callback,
            qos_profile
        )

    def _setup_publishers(self):
        """Set up publisher nodes."""
        self.velocity_pub = self.create_publisher(
            Marker,
            '/drone/velocity',
            10
        )
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/drone/pose_with_covariance',
            10
        )
        self.vehicle_path_pub = self.create_publisher(
            Path,
            "/drone/vehicle_path",
            10
        )
        self.setpoint_path_pub = self.create_publisher(
            Path,
            "/drone/setpoint_path",
            10
        )

    def create_velocity_marker(self, frame_id, id):
        """Create an arrow marker for velocity visualization."""
        msg = Marker()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "velocity"
        msg.id = id
        msg.type = Marker.ARROW
        msg.action = Marker.ADD

        # Arrow appearance
        msg.scale.x = 0.1  # shaft diameter
        msg.scale.y = 0.2  # head diameter
        msg.scale.z = 0.0
        msg.color.r = 0.5
        msg.color.g = 0.5
        msg.color.b = 0.0
        msg.color.a = 1.0

        # Arrow points
        dt = 0.5  # Scale factor for velocity vector
        tail_point = Point(
            x=float(self.vehicle_odometry.position[0]),
            y=float(-self.vehicle_odometry.position[1]),
            z=float(-self.vehicle_odometry.position[2])
        )
        head_point = Point(
            x=tail_point.x + dt * float(self.vehicle_odometry.velocity[0]),
            y=tail_point.y + dt * float(-self.vehicle_odometry.velocity[1]),
            z=tail_point.z + dt * float(-self.vehicle_odometry.velocity[2])
        )
        msg.points = [tail_point, head_point]
        return msg

    def create_pose_with_covariance(self, frame_id):
        """Convert VehicleOdometry to PoseWithCovarianceStamped message."""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id

        # Set position (NED to ENU conversion)
        pose_msg.pose.pose.position.x = float(
            self.vehicle_odometry.position[0])
        pose_msg.pose.pose.position.y = float(
            -self.vehicle_odometry.position[1])
        pose_msg.pose.pose.position.z = float(
            -self.vehicle_odometry.position[2])

        # Set orientation (NED to ENU conversion)
        pose_msg.pose.pose.orientation.w = float(self.vehicle_odometry.q[0])
        pose_msg.pose.pose.orientation.x = float(self.vehicle_odometry.q[1])
        pose_msg.pose.pose.orientation.y = float(-self.vehicle_odometry.q[2])
        pose_msg.pose.pose.orientation.z = float(-self.vehicle_odometry.q[3])

        # Set covariance matrix [x, y, z, roll, pitch, yaw]
        covariance = np.zeros(36)
        covariance[0] = float(self.vehicle_odometry.position_variance[0])
        covariance[7] = float(self.vehicle_odometry.position_variance[1])
        covariance[14] = float(self.vehicle_odometry.position_variance[2])
        covariance[21] = float(self.vehicle_odometry.orientation_variance[0])
        covariance[28] = float(self.vehicle_odometry.orientation_variance[1])
        covariance[35] = float(self.vehicle_odometry.orientation_variance[2])
        pose_msg.pose.covariance = covariance.tolist()

        return pose_msg

    def trajectory_setpoint_to_pose(self, frame_id):
        """Convert trajectory setpoint to PoseStamped message."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id

        # Set position
        pose_msg.pose.position.x = float(self.setpoint_position[0])
        pose_msg.pose.position.y = float(self.setpoint_position[1])
        pose_msg.pose.position.z = float(self.setpoint_position[2])

        # Use current vehicle orientation
        pose_msg.pose.orientation.w = float(self.vehicle_odometry.q[0])
        pose_msg.pose.orientation.x = float(self.vehicle_odometry.q[1])
        pose_msg.pose.orientation.y = float(-self.vehicle_odometry.q[2])
        pose_msg.pose.orientation.z = float(-self.vehicle_odometry.q[3])

        return pose_msg

    def append_vehicle_path(self, msg):
        """Append pose to vehicle path and maintain trail size."""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.vehicle_path_msg.poses.append(pose_stamped)
        if len(self.vehicle_path_msg.poses) > self.trail_size:
            del self.vehicle_path_msg.poses[0]

    def append_setpoint_path(self, msg):
        """Append pose to setpoint path and maintain trail size."""
        self.setpoint_path_msg.poses.append(msg)
        if len(self.setpoint_path_msg.poses) > self.trail_size:
            del self.setpoint_path_msg.poses[0]

    def cmdloop_callback(self):
        """Timer callback to publish visualization messages."""
        # Publish vehicle pose and path
        vehicle_pose_msg = self.create_pose_with_covariance("map")
        self.pose_pub.publish(vehicle_pose_msg)
        self.vehicle_path_msg.header = vehicle_pose_msg.header
        self.append_vehicle_path(vehicle_pose_msg)
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        # Publish setpoint path
        setpoint_pose_msg = self.trajectory_setpoint_to_pose("map")
        self.setpoint_path_msg.header = vehicle_pose_msg.header
        self.append_setpoint_path(setpoint_pose_msg)
        self.setpoint_path_pub.publish(self.setpoint_path_msg)

        # Publish velocity marker
        velocity_msg = self.create_velocity_marker("map", 1)
        self.velocity_pub.publish(velocity_msg)

    def trajectory_setpoint_callback(self, msg):
        """Callback for trajectory setpoint."""
        self.setpoint_position = np.array([
            msg.position[0],
            -msg.position[1],  # NED to ENU conversion
            -msg.position[2]   # NED to ENU conversion
        ])

    def odom_callback(self, msg):
        """Callback for vehicle odometry."""
        # Check path clearing timeout
        path_clearing_timeout = self.get_parameter(
            "path_clearing_timeout").get_parameter_value().double_value
        if path_clearing_timeout >= 0 and (
            (Clock().now().nanoseconds / 1e9 -
             self.last_odom_update) > path_clearing_timeout
        ):
            self.vehicle_path_msg.poses.clear()

        self.last_odom_update = Clock().now().nanoseconds / 1e9
        self.vehicle_odometry = msg


def main():
    rclpy.init()
    pose_visualizer = PoseVisualizer()
    try:
        rclpy.spin(pose_visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        pose_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

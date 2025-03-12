#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from px4_msgs.msg import VehicleOdometry, TrajectorySetpoint
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path, Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from rosgraph_msgs.msg import Clock as ClockMsg
from builtin_interfaces.msg import Time
import tf2_ros

class Visualizer(Node):
    """
    A ROS2 node that visualizes the drone's pose, path, and velocity.
    """

    def __init__(self):
        super().__init__('visualizer')

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
        
        # Initialize simulation clock
        self.sim_time = None
        
        # Subscribe to /clock topic for simulation time
        self.clock_sub = self.create_subscription(
            ClockMsg,
            '/clock',
            self.clock_callback,
            10
        )

        # Path visualization parameters
        self.trail_size = 1000
        self.vehicle_path_msg = Path()
        self.vehicle_path_msg.header.frame_id = "map"
        self.setpoint_path_msg = Path()
        self.setpoint_path_msg.header.frame_id = "map"
        
        # Initialize TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create subscribers
        self._setup_subscribers(qos_profile)

        # Create publishers
        self._setup_publishers()

        # Create timer for visualization updates
        self.create_timer(0.05, self.cmdloop_callback)  # 20Hz update rate

        # Parameters
        self.declare_parameter("path_clearing_timeout", -1.0)

        self.get_logger().info('Visualizer node initialized')

    def clock_callback(self, msg):
        """Callback for simulation clock updates."""
        self.sim_time = msg
        
    def get_current_timestamp(self):
        """Get current timestamp from sim clock if available, otherwise use system time."""
        if self.sim_time is not None:
            # Create a new Time message
            time_msg = Time()
            time_msg.sec = self.sim_time.clock.sec
            time_msg.nanosec = self.sim_time.clock.nanosec
            return time_msg
        else:
            # Use system time as fallback
            return self.get_clock().now().to_msg()

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
        # Add odometry publisher
        self.odom_pub = self.create_publisher(
            Odometry,
            '/drone/odom',
            10
        )

    def create_velocity_marker(self, frame_id, id):
        """Create an arrow marker for velocity visualization."""
        msg = Marker()
        msg.header.frame_id = frame_id
        # Timestamp will be set by the caller
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
        # Timestamp will be set by the caller
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
        covariance[0] = max(float(self.vehicle_odometry.position_variance[0]), 1e-5)
        covariance[7] = max(float(self.vehicle_odometry.position_variance[1]), 1e-5)
        covariance[14] = max(float(self.vehicle_odometry.position_variance[2]), 1e-5)
        covariance[21] = max(float(self.vehicle_odometry.orientation_variance[0]), 1e-5)
        covariance[28] = max(float(self.vehicle_odometry.orientation_variance[1]), 1e-5)
        covariance[35] = max(float(self.vehicle_odometry.orientation_variance[2]), 1e-5)
        pose_msg.pose.covariance = covariance.tolist()

        return pose_msg

    def trajectory_setpoint_to_pose(self, frame_id):
        """Convert trajectory setpoint to PoseStamped message."""
        pose_msg = PoseStamped()
        # Timestamp will be set by the caller
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

    def create_odom_message(self, frame_id, child_frame_id):
        """Convert VehicleOdometry to ROS Odometry message."""
        odom_msg = Odometry()
        # Timestamp will be set by the caller
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = child_frame_id
        
        # Set position (NED to ENU conversion)
        odom_msg.pose.pose.position.x = float(self.vehicle_odometry.position[0])
        odom_msg.pose.pose.position.y = float(-self.vehicle_odometry.position[1])
        odom_msg.pose.pose.position.z = float(-self.vehicle_odometry.position[2])
        
        # Set orientation (NED to ENU conversion)
        odom_msg.pose.pose.orientation.w = float(self.vehicle_odometry.q[0])
        odom_msg.pose.pose.orientation.x = float(self.vehicle_odometry.q[1])
        odom_msg.pose.pose.orientation.y = float(-self.vehicle_odometry.q[2])
        odom_msg.pose.pose.orientation.z = float(-self.vehicle_odometry.q[3])
        
        # Set velocity (NED to ENU conversion)
        odom_msg.twist.twist.linear.x = float(self.vehicle_odometry.velocity[0])
        odom_msg.twist.twist.linear.y = float(-self.vehicle_odometry.velocity[1])
        odom_msg.twist.twist.linear.z = float(-self.vehicle_odometry.velocity[2])
        
        # Set angular velocity (NED to ENU conversion)
        odom_msg.twist.twist.angular.x = float(self.vehicle_odometry.angular_velocity[0])
        odom_msg.twist.twist.angular.y = float(-self.vehicle_odometry.angular_velocity[1])
        odom_msg.twist.twist.angular.z = float(-self.vehicle_odometry.angular_velocity[2])
        
        # Set covariance matrices
        pose_covariance = np.zeros(36)
        pose_covariance[0] = max(float(self.vehicle_odometry.position_variance[0]), 1e-5)
        pose_covariance[7] = max(float(self.vehicle_odometry.position_variance[1]), 1e-5)
        pose_covariance[14] = max(float(self.vehicle_odometry.position_variance[2]), 1e-5)
        pose_covariance[21] = max(float(self.vehicle_odometry.orientation_variance[0]), 1e-5)
        pose_covariance[28] = max(float(self.vehicle_odometry.orientation_variance[1]), 1e-5)
        pose_covariance[35] = max(float(self.vehicle_odometry.orientation_variance[2]), 1e-5)

        odom_msg.pose.covariance = pose_covariance.tolist()
        
        # Set twist covariance (using velocity variance if available)
        twist_covariance = np.zeros(36)
        if hasattr(self.vehicle_odometry, 'velocity_variance'):
            twist_covariance[0] = max(float(self.vehicle_odometry.velocity_variance[0]), 1e-5)
            twist_covariance[7] = max(float(self.vehicle_odometry.velocity_variance[1]), 1e-5)
            twist_covariance[14] = max(float(self.vehicle_odometry.velocity_variance[2]), 1e-5)
            twist_covariance[21] = 1.0
            twist_covariance[28] = 1.0
            twist_covariance[35] = 1.0
        odom_msg.twist.covariance = twist_covariance.tolist()
        
        return odom_msg
        
    def create_transform_message(self, frame_id, child_frame_id):
        """Create transform message from odometry data."""
        transform = TransformStamped()
        # Timestamp will be set by the caller
        transform.header.frame_id = frame_id
        transform.child_frame_id = child_frame_id
        
        # Set translation (NED to ENU conversion)
        transform.transform.translation.x = float(self.vehicle_odometry.position[0])
        transform.transform.translation.y = float(-self.vehicle_odometry.position[1])
        transform.transform.translation.z = float(-self.vehicle_odometry.position[2])
        
        # Set rotation (NED to ENU conversion)
        transform.transform.rotation.w = float(self.vehicle_odometry.q[0])
        transform.transform.rotation.x = float(self.vehicle_odometry.q[1])
        transform.transform.rotation.y = float(-self.vehicle_odometry.q[2])
        transform.transform.rotation.z = float(-self.vehicle_odometry.q[3])
        
        return transform

    def cmdloop_callback(self):
        """Timer callback to publish visualization messages."""
        # Skip if we haven't received any clock messages yet
        if self.sim_time is None:
            return
            
        # Get current timestamp once to ensure all messages use the same timestamp
        current_timestamp = self.get_current_timestamp()
        
        # Publish vehicle pose and path
        vehicle_pose_msg = self.create_pose_with_covariance("map")
        vehicle_pose_msg.header.stamp = current_timestamp
        self.pose_pub.publish(vehicle_pose_msg)
        
        self.vehicle_path_msg.header.stamp = current_timestamp
        self.append_vehicle_path(vehicle_pose_msg)
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        # Publish setpoint path
        setpoint_pose_msg = self.trajectory_setpoint_to_pose("map")
        setpoint_pose_msg.header.stamp = current_timestamp
        self.setpoint_path_msg.header.stamp = current_timestamp
        self.append_setpoint_path(setpoint_pose_msg)
        self.setpoint_path_pub.publish(self.setpoint_path_msg)

        # Publish velocity marker
        velocity_msg = self.create_velocity_marker("map", 1)
        velocity_msg.header.stamp = current_timestamp
        self.velocity_pub.publish(velocity_msg)
        
        # Publish odometry message
        odom_msg = self.create_odom_message("odom", "base_link")
        odom_msg.header.stamp = current_timestamp
        self.odom_pub.publish(odom_msg)
        
        # Publish TF transform
        transform_msg = self.create_transform_message("odom", "base_link")
        transform_msg.header.stamp = current_timestamp
        self.tf_broadcaster.sendTransform(transform_msg)
        
        # Also publish map to odom transform (identity transform)
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = current_timestamp
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0  # Identity quaternion
        self.tf_broadcaster.sendTransform(map_to_odom)

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
    visualizer = Visualizer()
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

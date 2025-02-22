#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time

from px4_msgs.msg import VehicleOdometry, OffboardControlMode, VehicleCommand, VehicleStatus, TrajectorySetpoint
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration


class PX4LawnMowerController(Node):
    """
    A ROS 2 node for controlling PX4 in a lawn mower pattern using offboard control.
    """

    def __init__(self):
        super().__init__('px4_lawn_mower_controller')

        # Configure QoS profile for PX4 communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Set up publishers and subscribers
        self._setup_publishers(qos_profile)
        self._setup_subscribers(qos_profile)

        # Mission parameters
        self.TARGET_HEIGHT = 3.5        # Target flight height in meters
        self.RECTANGLE_WIDTH = 7.0      # Width of search area in meters
        self.RECTANGLE_LENGTH = 3.0     # Length of search area in meters
        self.LANE_SPACING = 1.0         # Distance between parallel lanes in meters
        self.MAX_VELOCITY = 5.0         # Maximum velocity for all movements in m/s
        self.LANDING_VELOCITY = 2.0     # Landing descent rate in m/s

        # Control thresholds
        self.HEIGHT_REACHED_THRESHOLD = 0.3  # Tolerance for height control in meters
        self.POSITION_THRESHOLD = 0.3        # Tolerance for position control in meters
        # Height above ground to consider landed in meters
        self.GROUND_THRESHOLD = 0.05

        # Initialize state machine
        self.state = "TAKEOFF"          # Current state of the mission
        self.mission_completed = False   # Flag for mission completion
        self.landing_completed = False   # Flag for landing completion

        # Initialize PX4 control flags
        self.offboard_mode_enabled = False
        self.arm_state = False
        self.offboard_setpoint_counter = 0

        # Initialize position tracking
        self.pattern_origin_x = 0.0     # X coordinate of takeoff position
        self.pattern_origin_y = 0.0     # Y coordinate of takeoff position
        self.initial_yaw = 0.0          # Initial heading of the drone
        self.initial_yaw_set = False
        self.rotation_matrix = None      # For rotating pattern based on initial heading

        # Initialize waypoint control
        self.current_waypoint_index = 0
        self.waypoints = []             # List of [x, y] waypoints for the pattern

        # Set up visualization
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/waypoint_markers',
            10
        )

        # Create control timer (10Hz)
        self.create_timer(0.1, self.timer_callback)

    def _setup_publishers(self, qos_profile):
        """Set up all publishers with specified QoS profile."""
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

    def _setup_subscribers(self, qos_profile):
        """Set up all subscribers with specified QoS profile."""
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback, qos_profile)

        # Initialize message storage
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()

    def vehicle_odometry_callback(self, msg):
        """Callback function for vehicle odometry data."""
        self.vehicle_odometry = msg

        # Capture initial yaw angle if not set
        if not self.initial_yaw_set and hasattr(msg, 'q'):
            # Convert quaternion to Euler angles
            roll, pitch, yaw = self.quaternion_to_euler(msg.q)
            self.initial_yaw = yaw
            self.initial_yaw_set = True

    def vehicle_status_callback(self, msg):
        """Callback function for vehicle status data."""
        self.vehicle_status = msg

        # Update status flags
        self.offboard_mode_enabled = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.arm_state = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED

        # Log state changes
        self.get_logger().debug(
            f"Vehicle Status - Offboard: {self.offboard_mode_enabled}, Armed: {self.arm_state}")

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arm command sent")

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info("Disarm command sent")

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Offboard mode command sent")

    def publish_offboard_control_mode(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        """Publish trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [float(p) for p in [x, y, z]]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Publish vehicle command."""
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def is_at_target_height(self):
        """Check if the drone has reached the target height."""
        current_height = -self.vehicle_odometry.position[2]
        return abs(current_height - self.TARGET_HEIGHT) < self.HEIGHT_REACHED_THRESHOLD

    def calculate_limited_position(self, current_x, current_y, target_x, target_y, max_velocity):
        """Calculate next position with constant velocity."""
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance <= 0.05:  # Very close to target
            return target_x, target_y, 0.0

        # Calculate unit vector of movement direction
        dx_unit = dx / distance
        dy_unit = dy / distance

        # Calculate step size based on velocity and control period
        step_size = max_velocity * 0.1  # 0.1 is the control period

        # Calculate next position along the straight line
        new_x = current_x + dx_unit * step_size
        new_y = current_y + dy_unit * step_size

        return new_x, new_y, max_velocity

    def set_rotation_matrix(self):
        """Set up rotation matrix based on initial yaw."""
        cos_yaw = math.cos(self.initial_yaw)
        sin_yaw = math.sin(self.initial_yaw)
        self.rotation_matrix = {
            'cos_yaw': cos_yaw,
            'sin_yaw': sin_yaw
        }

    def rotate_point(self, x, y):
        """Rotate point around origin based on initial yaw."""
        # Translate point relative to pattern origin
        x_rel = x - self.pattern_origin_x
        y_rel = y - self.pattern_origin_y

        # Rotate point
        x_rot = x_rel * \
            self.rotation_matrix['cos_yaw'] - \
            y_rel * self.rotation_matrix['sin_yaw']
        y_rot = x_rel * \
            self.rotation_matrix['sin_yaw'] + \
            y_rel * self.rotation_matrix['cos_yaw']

        # Translate back
        x_final = x_rot + self.pattern_origin_x
        y_final = y_rot + self.pattern_origin_y

        return x_final, y_final

    def generate_lawn_mower_waypoints(self):
        """Generate all waypoints for the lawn mower pattern."""
        waypoints = []
        num_lanes = int(self.RECTANGLE_WIDTH // self.LANE_SPACING)
        flag = self.RECTANGLE_WIDTH % self.LANE_SPACING > 0

        if flag:
            num_lanes += 1
            for i in range(num_lanes):
                if i == num_lanes - 1:
                    x = self.pattern_origin_x + self.RECTANGLE_WIDTH
                else:
                    x = self.pattern_origin_x + ((i + 1) * self.LANE_SPACING)
                y = self.pattern_origin_y
                if i % 2 == 0:  # Add length on even indices
                    y += self.RECTANGLE_LENGTH
                waypoints.append(self.rotate_point(x, y))
        else:
            for i in range(num_lanes):
                x = self.pattern_origin_x + ((i + 1) * self.LANE_SPACING)
                y = self.pattern_origin_y
                if i % 2 == 0:  # Add length on even indices
                    y += self.RECTANGLE_LENGTH
                waypoints.append(self.rotate_point(x, y))

        # Add final return to start point
        waypoints.append([self.pattern_origin_x, self.pattern_origin_y])
        return waypoints

    def calculate_lawn_mower_setpoint(self):
        """Calculate the next setpoint between current waypoints."""
        if self.current_waypoint_index >= len(self.waypoints):
            self.mission_completed = True
            return self.pattern_origin_x, self.pattern_origin_y, 0.0

        target_x, target_y = self.waypoints[self.current_waypoint_index]

        # Calculate next position with velocity control
        next_x, next_y, current_velocity = self.calculate_limited_position(
            self.vehicle_odometry.position[0],
            self.vehicle_odometry.position[1],
            target_x,
            target_y,
            self.MAX_VELOCITY
        )

        # Check if waypoint is reached
        distance_to_target = math.sqrt(
            (self.vehicle_odometry.position[0] - target_x) ** 2 +
            (self.vehicle_odometry.position[1] - target_y) ** 2
        )

        if distance_to_target < self.POSITION_THRESHOLD:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.mission_completed = True

        return next_x, next_y, current_velocity

    def is_at_target_position(self, target_x, target_y):
        """Check if drone has reached target position within threshold."""
        current_x = self.vehicle_odometry.position[0]
        current_y = self.vehicle_odometry.position[1]
        distance = math.sqrt((current_x - target_x)**2 +
                             (current_y - target_y)**2)
        return distance < self.POSITION_THRESHOLD

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Extract quaternion components
        w, x, y, z = q[0], q[1], q[2], q[3]

        # Calculate yaw (z-axis rotation)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Calculate pitch (y-axis rotation)
        pitch = math.asin(2.0 * (w * y - z * x))

        # Calculate roll (x-axis rotation)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

        return roll, pitch, yaw

    def publish_waypoint_markers(self):
        """Publish waypoint markers for visualization in RViz2."""
        marker_array = MarkerArray()
        for i in range(len(self.waypoints)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set lifetime to 0 for persistent markers
            marker.lifetime = Duration(seconds=0).to_msg()

            # Set the scale of the markers
            marker.pose.position.x = float(self.waypoints[i][0])
            marker.pose.position.y = float(-self.waypoints[i][1])
            marker.pose.position.z = float(self.TARGET_HEIGHT)
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            # Set the color (blue)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

    def calculate_limited_height(self, current_height, target_height, max_velocity):
        """Calculate next height with velocity control."""
        height_error = target_height - current_height

        # Calculate step size based on velocity and control period
        step_size = max_velocity * 0.1  # 0.1 is control period

        # Limit height change by velocity
        if abs(height_error) <= step_size:
            return target_height
        else:
            if height_error > 0:
                return current_height + step_size
            else:
                return current_height - step_size

    def timer_callback(self):
        """Timer callback for control loop."""
        # Always publish offboard control mode
        self.publish_offboard_control_mode()

        # Start offboard control and arming sequence after 10 setpoints
        if self.offboard_setpoint_counter >= 10:
            # Only proceed if we haven't completed landing
            if not self.landing_completed:
                # Check if offboard mode is enabled
                if not self.offboard_mode_enabled:
                    self.get_logger().info("Waiting for offboard mode...")
                    self.engage_offboard_mode()
                # Only try to arm once offboard mode is confirmed
                elif not self.arm_state:
                    self.get_logger().info("Offboard mode confirmed, attempting to arm...")
                    self.arm()
                    self.start_time = time.time()
                # Only proceed with trajectory once armed
                elif self.arm_state:
                    if self.state == "TAKEOFF":
                        if not self.is_at_target_height():
                            current_height = -self.vehicle_odometry.position[2]
                            next_height = self.calculate_limited_height(
                                current_height,
                                self.TARGET_HEIGHT,
                                self.MAX_VELOCITY
                            )

                            self.publish_trajectory_setpoint(
                                x=self.vehicle_odometry.position[0],
                                y=self.vehicle_odometry.position[1],
                                z=-next_height,
                                yaw=self.initial_yaw
                            )
                            self.get_logger().info(
                                f"Taking off... Current height: {current_height:.2f}m, "
                                f"Target height: {self.TARGET_HEIGHT:.2f}m, "
                                f"Velocity: {self.MAX_VELOCITY:.2f}m/s"
                            )
                        else:
                            # Set pattern origin to takeoff position
                            self.pattern_origin_x = self.vehicle_odometry.position[0]
                            self.pattern_origin_y = self.vehicle_odometry.position[1]
                            # Set up rotation matrix based on initial yaw
                            self.set_rotation_matrix()
                            # Generate waypoints for the entire pattern
                            self.waypoints = self.generate_lawn_mower_waypoints()
                            # Publish waypoint markers
                            self.publish_waypoint_markers()
                            self.state = "LAWN_MOWER"
                            self.get_logger().info(
                                f"Starting lawn mower pattern from ({self.pattern_origin_x:.2f}, {self.pattern_origin_y:.2f})"
                                f" with heading {math.degrees(self.initial_yaw):.1f}°"
                            )

                    elif self.state == "LAWN_MOWER":
                        x, y, velocity = self.calculate_lawn_mower_setpoint()
                        self.publish_trajectory_setpoint(
                            x=x,
                            y=y,
                            z=-self.TARGET_HEIGHT,
                            yaw=self.initial_yaw
                        )

                        if not self.mission_completed:
                            current_waypoint = self.waypoints[self.current_waypoint_index]
                            self.get_logger().info(
                                f"Following lawn mower pattern: "
                                f"Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}, "
                                f"Current: ({self.vehicle_odometry.position[0]:.2f}, {self.vehicle_odometry.position[1]:.2f}), "
                                f"Target: ({current_waypoint[0]:.2f}, {current_waypoint[1]:.2f}), "
                                f"Velocity: {velocity:.2f}m/s"
                            )

                        else:
                            # Check if we've reached the start position before landing
                            if self.is_at_target_position(self.pattern_origin_x, self.pattern_origin_y):
                                self.state = "LANDING"
                                self.get_logger().info("Pattern completed, starting landing sequence")
                            else:
                                self.get_logger().info(
                                    f"Returning to start position: "
                                    f"Current: ({self.vehicle_odometry.position[0]:.2f}, {self.vehicle_odometry.position[1]:.2f}), "
                                    f"Target: ({self.pattern_origin_x:.2f}, {self.pattern_origin_y:.2f}), "
                                    f"Velocity: {velocity:.2f}m/s"
                                )

                    elif self.state == "LANDING":
                        current_height = -self.vehicle_odometry.position[2]

                        if current_height > self.GROUND_THRESHOLD:
                            # Use calculate_limited_height for smooth landing
                            next_height = self.calculate_limited_height(
                                current_height,
                                0.0,  # Target height is 0
                                self.LANDING_VELOCITY
                            )

                            self.publish_trajectory_setpoint(
                                x=self.pattern_origin_x,
                                y=self.pattern_origin_y,
                                z=-next_height,
                                yaw=self.initial_yaw
                            )

                            self.get_logger().info(
                                f"Landing... Height: {current_height:.2f}m, "
                                f"Target Height: {next_height:.2f}m, "
                                f"Descent rate: {self.LANDING_VELOCITY:.1f}m/s"
                            )
                        else:
                            # When close to ground, disarm and set landing completed
                            self.disarm()
                            self.landing_completed = True
                            self.state = "MISSION_COMPLETE"
                            self.get_logger().info("Landing completed, mission ended")
            else:
                self.disarm()

        self.offboard_setpoint_counter += 1


def main(args=None):
    rclpy.init(args=args)
    controller = PX4LawnMowerController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Stopping controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

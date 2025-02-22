#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time

from px4_msgs.msg import VehicleOdometry, OffboardControlMode, VehicleCommand, VehicleStatus, TrajectorySetpoint

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

        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Subscribers
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.start_time = time.time()
        
        # Flight parameters
        self.TARGET_HEIGHT = 3.5  # meters
        self.RECTANGLE_WIDTH = 7.0  # meters
        self.RECTANGLE_LENGTH = 3.0  # meters
        self.LANE_SPACING = 1.5  # meters between parallel lanes
        self.HEIGHT_REACHED_THRESHOLD = 0.3  # meters
        self.POSITION_THRESHOLD = 0.5  # meters
        
        # Velocity control parameter
        self.MAX_VELOCITY = 5.0  # m/s for pattern following
        
        # Pattern direction (front and right from takeoff point)
        self.pattern_origin_x = 0.0  # Will be set at takeoff
        self.pattern_origin_y = 0.0  # Will be set at takeoff

        # State machine
        self.state = "TAKEOFF"  # States: TAKEOFF, LAWN_MOWER, RETURN_TO_START, LANDING, MISSION_COMPLETE
        self.current_lane = 0
        self.moving_right = True
        self.mission_completed = False
        self.pattern_started = False
        self.landing_completed = False  # Add flag to track landing completion
        
        # Control parameters
        self.height_P_gain = 2.0
        self.max_vertical_velocity = 2.0
        self.takeoff_velocity = 1.5

        # Add status flags
        self.offboard_mode_enabled = False
        self.arm_state = False

        # Add initial yaw tracking
        self.initial_yaw = 0.0
        self.initial_yaw_set = False

        # Add rotation matrix components
        self.rotation_matrix = None  # Will be set after getting initial yaw

        # Landing parameters
        self.LANDING_VELOCITY = 1.2  # m/s for landing
        self.GROUND_THRESHOLD = 0.1  # meters above ground to consider landed

        # Create a timer to publish control commands
        self.create_timer(0.1, self.timer_callback)  # 10Hz control loop

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
        self.get_logger().debug(f"Vehicle Status - Offboard: {self.offboard_mode_enabled}, Armed: {self.arm_state}")

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
            
        # Limit step size based on velocity and control period (0.1s)
        max_step = max_velocity * 0.1
        
        # Scale the movement to respect max velocity
        scale = max_step / distance
        new_x = current_x + dx * scale
        new_y = current_y + dy * scale
        
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
        x_rot = x_rel * self.rotation_matrix['cos_yaw'] - y_rel * self.rotation_matrix['sin_yaw']
        y_rot = x_rel * self.rotation_matrix['sin_yaw'] + y_rel * self.rotation_matrix['cos_yaw']
        
        # Translate back
        x_final = x_rot + self.pattern_origin_x
        y_final = y_rot + self.pattern_origin_y
        
        return x_final, y_final

    def calculate_lawn_mower_setpoint(self):
        """Calculate the next setpoint in the lawn mower pattern."""
        if self.mission_completed:
            next_x, next_y, current_velocity = self.calculate_limited_position(
                self.vehicle_odometry.position[0],
                self.vehicle_odometry.position[1],
                self.pattern_origin_x,
                self.pattern_origin_y,
                self.MAX_VELOCITY
            )
            
            if self.is_at_target_position(self.pattern_origin_x, self.pattern_origin_y):
                self.state = "LANDING"
                self.get_logger().info("Pattern completed, starting landing sequence")
                
            return next_x, next_y, current_velocity

        # Calculate unrotated pattern points
        x = self.pattern_origin_x + (self.current_lane * self.LANE_SPACING)
        
        # If pattern hasn't started, begin with moving to first lane
        if not self.pattern_started:
            y = self.pattern_origin_y
            self.pattern_started = True
        else:
            if self.moving_right:
                y = self.pattern_origin_y + self.RECTANGLE_LENGTH
            else:
                y = self.pattern_origin_y
            
        # Rotate the target point based on initial heading
        target_x, target_y = self.rotate_point(x, y)
            
        # Calculate next position with velocity control
        next_x, next_y, current_velocity = self.calculate_limited_position(
            self.vehicle_odometry.position[0],
            self.vehicle_odometry.position[1],
            target_x,
            target_y,
            self.MAX_VELOCITY
        )

        # Check if we need to switch lanes - using distance to target
        distance_to_target = math.sqrt(
            (self.vehicle_odometry.position[0] - target_x) ** 2 +
            (self.vehicle_odometry.position[1] - target_y) ** 2
        )
            
        # Check if we need to switch lanes
        if distance_to_target < self.POSITION_THRESHOLD:
            if self.moving_right:
                # Reached top, move back down
                self.moving_right = False
            else:
                # Reached bottom, move to next lane
                self.moving_right = True
                self.current_lane += 1
            
        # Check if we've completed the pattern
        if self.current_lane * self.LANE_SPACING > self.RECTANGLE_WIDTH:
            self.mission_completed = True
            
        return next_x, next_y, current_velocity

    def is_at_target_position(self, target_x, target_y):
        """Check if drone has reached target position within threshold."""
        current_x = self.vehicle_odometry.position[0]
        current_y = self.vehicle_odometry.position[1]
        distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
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
                            self.publish_trajectory_setpoint(
                                x=self.vehicle_odometry.position[0],
                                y=self.vehicle_odometry.position[1],
                                z=-self.TARGET_HEIGHT,
                                yaw=self.initial_yaw
                            )
                            self.get_logger().info(f"Taking off... Current height: {-self.vehicle_odometry.position[2]:.2f}m")
                        else:
                            # Set pattern origin to takeoff position
                            self.pattern_origin_x = self.vehicle_odometry.position[0]
                            self.pattern_origin_y = self.vehicle_odometry.position[1]
                            # Set up rotation matrix based on initial yaw
                            self.set_rotation_matrix()
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
                        
                        if self.mission_completed:
                            self.get_logger().info(
                                f"Returning to start: "
                                f"Current: ({self.vehicle_odometry.position[0]:.2f}, {self.vehicle_odometry.position[1]:.2f}), "
                                f"Target: ({self.pattern_origin_x:.2f}, {self.pattern_origin_y:.2f}), "
                                f"Velocity: {velocity:.2f}m/s"
                            )
                        else:
                            # Calculate unrotated pattern points for logging
                            unrotated_x = self.pattern_origin_x + (self.current_lane * self.LANE_SPACING)
                            unrotated_y = self.pattern_origin_y + (self.RECTANGLE_LENGTH if self.moving_right else 0.0)
                            
                            # Get rotated target for logging
                            target_x, target_y = self.rotate_point(unrotated_x, unrotated_y)
                            
                            self.get_logger().info(
                                f"Following lawn mower pattern: "
                                f"Lane {self.current_lane}, "
                                f"Current: ({self.vehicle_odometry.position[0]:.2f}, {self.vehicle_odometry.position[1]:.2f}), "
                                f"Target: ({target_x:.2f}, {target_y:.2f}), "
                                f"Direction: {'Up' if self.moving_right else 'Down'}, "
                                f"Velocity: {velocity:.2f}m/s"
                            )

                    elif self.state == "LANDING":
                        current_height = -self.vehicle_odometry.position[2]
                        
                        if current_height > self.GROUND_THRESHOLD:
                            # Calculate landing setpoint with constant descent rate
                            descent_distance = self.LANDING_VELOCITY * 0.1  # 0.1 is control period
                            target_height = max(0.0, current_height - descent_distance)
                            
                            self.publish_trajectory_setpoint(
                                x=self.pattern_origin_x,
                                y=self.pattern_origin_y,
                                z=-target_height,
                                yaw=self.initial_yaw
                            )
                            
                            self.get_logger().info(
                                f"Landing... Height: {current_height:.2f}m, "
                                f"Descent rate: {self.LANDING_VELOCITY:.1f}m/s"
                            )
                        else:
                            # When close to ground, disarm and set landing completed
                            self.disarm()
                            self.landing_completed = True
                            self.state = "MISSION_COMPLETE"
                            self.get_logger().info("Landing completed, mission ended")

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
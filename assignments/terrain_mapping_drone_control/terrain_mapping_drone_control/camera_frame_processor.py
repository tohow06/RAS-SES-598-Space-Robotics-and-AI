#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock as ClockMsg
from builtin_interfaces.msg import Time

class CameraFrameProcessor(Node):
    """
    A ROS2 node that:
    1. Resizes RGB images to match depth image dimensions
    2. Ensures all camera frames have the correct frame_id
    3. Synchronizes timestamps using simulation clock
    """

    def __init__(self):
        super().__init__('camera_frame_processor')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Initialize simulation clock
        self.sim_time = None
        
        # Subscribe to /clock topic for simulation time
        self.clock_sub = self.create_subscription(
            ClockMsg,
            '/clock',
            self.clock_callback,
            10
        )
        
        # Configure QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to RGB image
        self.rgb_sub = self.create_subscription(
            Image,
            '/drone/front_rgb',
            self.rgb_callback,
            qos_profile
        )
        
        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image,
            '/drone/front_depth',
            self.depth_callback,
            qos_profile
        )
        
        # Subscribe to camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/drone/front_rgb/camera_info',
            self.camera_info_callback,
            qos_profile
        )
        
        # Publishers
        self.rgb_processed_pub = self.create_publisher(
            Image,
            '/drone/front_rgb_processed',
            10
        )
        
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/drone/front_rgb_processed/camera_info',
            10
        )
        
        # Add depth republisher
        self.depth_republish_pub = self.create_publisher(
            Image,
            '/drone/front_depth_processed',
            10
        )
        
        # Store depth image dimensions
        self.depth_width = 640  # Default
        self.depth_height = 480  # Default
        self.have_depth_dims = False
        
        # Store camera info
        self.camera_info = None
        
        # Set the target frame_id for all camera messages
        self.target_frame_id = "base_link"
        
        self.get_logger().info('Camera frame processor node initialized')
        self.get_logger().info(f'All camera frames will be set to: {self.target_frame_id}')
        
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
        
    def depth_callback(self, msg):
        """Callback for depth image to get its dimensions and republish with correct frame_id."""
        if not self.have_depth_dims:
            self.depth_width = msg.width
            self.depth_height = msg.height
            self.have_depth_dims = True
            self.get_logger().info(f'Depth image dimensions: {self.depth_width}x{self.depth_height}')
        
        # Republish depth image with correct frame_id
        processed_msg = Image()
        processed_msg.header = msg.header
        processed_msg.header.frame_id = self.target_frame_id
        processed_msg.header.stamp = self.get_current_timestamp()
        processed_msg.height = msg.height
        processed_msg.width = msg.width
        processed_msg.encoding = msg.encoding
        processed_msg.is_bigendian = msg.is_bigendian
        processed_msg.step = msg.step
        processed_msg.data = msg.data
        
        self.depth_republish_pub.publish(processed_msg)
            
    def camera_info_callback(self, msg):
        """Callback for camera info."""
        self.camera_info = msg
        
    def rgb_callback(self, msg):
        """Callback for RGB image to resize it and set the correct frame_id."""
        # Skip processing if we don't have depth dimensions or camera info yet
        if not self.have_depth_dims or self.camera_info is None:
            return
            
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Resize image to match depth dimensions
            resized_image = cv2.resize(cv_image, (self.depth_width, self.depth_height), 
                                      interpolation=cv2.INTER_AREA)
            
            # Convert back to ROS Image
            processed_msg = self.cv_bridge.cv2_to_imgmsg(resized_image, encoding='bgr8')
            processed_msg.header = msg.header
            processed_msg.header.frame_id = self.target_frame_id
            processed_msg.header.stamp = self.get_current_timestamp()
            
            # Publish processed image
            self.rgb_processed_pub.publish(processed_msg)
            
            # Create updated camera info
            updated_camera_info = CameraInfo()
            updated_camera_info.header.frame_id = self.target_frame_id
            updated_camera_info.header.stamp = self.get_current_timestamp()
            
            # Scale camera intrinsics
            scale_x = float(self.depth_width) / float(msg.width)
            scale_y = float(self.depth_height) / float(msg.height)
            
            # Copy and scale the intrinsic matrix
            updated_camera_info.height = self.depth_height
            updated_camera_info.width = self.depth_width
            updated_camera_info.distortion_model = self.camera_info.distortion_model
            updated_camera_info.d = self.camera_info.d
            
            # Update intrinsic matrix (K)
            k = list(self.camera_info.k)
            k[0] *= scale_x  # fx
            k[2] *= scale_x  # cx
            k[4] *= scale_y  # fy
            k[5] *= scale_y  # cy
            updated_camera_info.k = tuple(k)
            
            # Update projection matrix (P)
            p = list(self.camera_info.p)
            p[0] *= scale_x  # fx
            p[2] *= scale_x  # cx
            p[5] *= scale_y  # fy
            p[6] *= scale_y  # cy
            updated_camera_info.p = tuple(p)
            
            # Copy rectification matrix (R)
            updated_camera_info.r = self.camera_info.r
            
            # Publish updated camera info
            self.camera_info_pub.publish(updated_camera_info)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main():
    rclpy.init()
    node = CameraFrameProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
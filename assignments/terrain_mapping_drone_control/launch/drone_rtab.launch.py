#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    """Generate launch description for RTAB-Map stereo SLAM."""

    remappings = [
        ('rgb/image', '/drone/front_rgb_processed'),
        ('depth/image', '/drone/front_depth_processed'),
        ('rgb/camera_info', '/drone/front_rgb_processed/camera_info'),
        ('odom', '/drone/odom')
    ]

    return LaunchDescription([
        # Camera frame processor node
        Node(
            package='terrain_mapping_drone_control',
            executable='camera_frame_processor',
            name='camera_frame_processor',
            output='screen'
        ),
        Node(
            package='rtabmap_sync', executable='rgbd_sync', output='screen',
            parameters=[{'approx_sync': True,
                         'sync_queue_size': 100,
                         'topic_queue_size': 10,
                         'approx_sync_max_interval': 0.01,
                         'use_sim_time': True}],
            remappings=remappings),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            arguments=['--delete_db_on_start'],
            parameters=[{
                        'subscribe_depth': False,
                        'subscribe_rgbd': True,
                        'frame_id': 'base_link',
                        'odom_frame_id': 'odom',
                        'use_sim_time': True,
                        'RGBD/AngularUpdate': '0.01',
                        'RGBD/LinearUpdate': '0.01',
                        'RGBD/OptimizeFromGraph': 'True',
                        }],
            remappings=remappings
        )
    ])

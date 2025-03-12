#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    """Generate launch description for RTAB-Map stereo SLAM."""

    return LaunchDescription([
        # Camera frame processor node
        Node(
            package='terrain_mapping_drone_control',
            executable='camera_frame_processor',
            name='camera_frame_processor',
            output='screen'
        ),

        # RTAB-Map Visual SLAM
        GroupAction(
            actions=[
                PushRosNamespace('rtabmap'),

                Node(
                    package='rtabmap_slam',
                    executable='rtabmap',
                    name='rtabmap',
                    output='screen',
                    arguments=['--delete_db_on_start'],
                    parameters=[{
                        'subscribe_depth': True,
                        'subscribe_laserScan': False,
                        'frame_id': 'base_link',
                        'queue_size': 2000,
                        'approx_sync': True,
                        'Vis/MinInliers': '12'
                    }], 
                    remappings=[
                        ('rgb/image', '/drone/front_rgb_processed'),
                        ('rgb/camera_info', '/drone/front_rgb_processed/camera_info'),
                        ('depth/image', '/drone/front_depth_processed'),
                        ('odom', '/drone/odom')
                    ]
                )
            ]
        )
    ])

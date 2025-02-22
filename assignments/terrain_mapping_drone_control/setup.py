from setuptools import setup
import os
from glob import glob

package_name = 'terrain_mapping_drone_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', [
            'config/terrain_mapping_params.yaml',
            'config/drone_viz.rviz'
        ]),
        ('share/' + package_name + '/models/terrain', [
            'models/terrain/model.config',
            'models/terrain/model.sdf'
        ]),
        ('share/' + package_name + '/models/terrain/meshes', [
            'models/terrain/meshes/artburysol175.obj',
            'models/terrain/meshes/artburysol175.mtl'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A ROS2 package for executing spiral trajectory with PX4',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'spiral_trajectory = terrain_mapping_drone_control.spiral_trajectory:main',
            'feature_tracker = terrain_mapping_drone_control.feature_tracker:main',
            'visualizer = terrain_mapping_drone_control.visualizer:main',
            'lawn_mower_trajectory_controller = terrain_mapping_drone_control.lawn_mower_trajectory_controller:main',
            'lawn_mower_trajectory_controllerV2 = terrain_mapping_drone_control.lawn_mower_trajectory_controller_V2_untest:main',
        ],
    },
) 
import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'piper_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('piper_control/configs/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'PyYAML',
        'python-can',
        'typing_extensions',
        'piper_sdk',
    ],
    zip_safe=True,
    maintainer='philip',
    maintainer_email='philipha@andrew.cmu.edu',
    description='Piper arm ROS2 hardware interface for FACTR2 demos.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'piper_node = piper_control.piper_node:main',
        ],
    },
)

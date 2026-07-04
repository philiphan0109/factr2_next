import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'teacher_arm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('teacher_arm/configs/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('teacher_arm/urdf/*.urdf')),
    ],
    install_requires=[
        'setuptools',
        'dynamixel-sdk==4.0.5',
    ],
    zip_safe=True,
    maintainer='philip',
    maintainer_email='philipha@andrew.cmu.edu',
    description='Teacher-arm teleoperation and optional FACTR2 feedback demo.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'piper_teleop = teacher_arm.piper_teleop:main',
        ],
    },
)

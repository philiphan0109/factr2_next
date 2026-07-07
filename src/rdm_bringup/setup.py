from setuptools import find_packages, setup
from glob import glob 
import os

package_name = 'rdm_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob("configs/*.yaml")),
        (os.path.join('share', package_name, 'launch'), glob("launch/*.launch.py"))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kshaw',
    maintainer_email='philiphan0109@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
